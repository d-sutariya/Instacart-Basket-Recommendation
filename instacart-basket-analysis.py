

import time
import pyspark
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, DoubleType


# How often user has reordered
class FeatureGenerator:
    
    def __init__(self,prior_product_orders,prior_orders_df,products_df):
        
        self.prior_product_orders = prior_product_orders
        self.prior_orders_df = prior_orders_df
        self.products_df= products_df


    def generate_user_related_features(self):
        
        df_with_num_of_reord = (
            self.prior_product_orders.select("reordered", "order_id")
            .join(self.prior_orders_df.select("user_id", "order_id"), how="left", on="order_id")
            .select("user_id", "reordered")
            .groupBy("user_id")
            .agg(F.count(F.col("reordered")).alias("frequency of reorder"))
        )
        
        # Time since previous order
        df_with_time_since_prev_ord = (
            self.prior_orders_df.select("user_id", "days_since_prior_order", "order_hour_of_day", "order_number", "order_id")
            .withColumn("privious_order_hour", F.lag("order_hour_of_day", 1).over(Window.partitionBy("user_id").orderBy("order_number")))
            .withColumn("time_since_Last_order", F.col("days_since_prior_order") * 24 + F.col("order_hour_of_day") - F.col("privious_order_hour"))
            .select("order_id", "time_since_last_order")
        )
        
        # Time of the day user visits
        df_with_time_of_day_usr_visits = (
            self.prior_orders_df.select("user_id", "order_hour_of_day", "order_id")
            .groupBy("user_id", "order_hour_of_day")
            .agg(F.count("order_id").alias("frequency"))
            .groupBy("user_id")
            .agg(F.max("frequency").alias("maximum_frquency"))
        )
        
        # Does the user order Asian, gluten-free, or organic items
        df_with_does_usr_asian_gluten_orga_items_ord = (
            self.prior_product_orders.select("order_id", "product_id")
            .join(self.products_df.select("product_id", "product_name"), on="product_id", how='left')
            .join(self.prior_orders_df.select("user_id", "order_id"), on="order_id", how='left')
            .groupBy("user_id", "order_id")
            .agg(F.collect_list("product_name").alias("list_of_products"))
            .withColumn("normalized_list", F.expr("transform(list_of_products, x -> lower(x))"))
            .withColumn("contains_or_not", 
                        F.expr("exists(normalized_list,x -> x like '%organic%')") |
                        F.expr("exists(normalized_list, x -> x like '%asian%')") |
                        F.expr("exists(normalized_list, x -> x like '%gluten free%')")
            )
            .select("order_id", "contains_or_not")
        )
        
        # Feature based on order size
        df_with_fets_of_ord_size = (
            self.prior_product_orders.select("product_id", "order_id")
            .join(self.prior_orders_df.select("user_id", "order_id"), on="order_id", how="left")
            .groupBy("user_id", 'order_id')
            .agg(F.count(F.col("product_id")).alias("count_of_product"))
            .groupBy("user_id")
            .agg(F.max(F.col("count_of_product")).alias("max_count_of_products"),
                 F.min(F.col("count_of_product")).alias("min_count_of_products"),
                 F.mean(F.col("count_of_product")).alias("mean_count_of_products"))
        )
        
        # How many of the user’s orders contained no previously purchased items
        df_with_freq_ord_that_hasnt_prev_purch_items = (
            self.prior_product_orders.select("order_id", "reordered")
            .join(self.prior_orders_df.select("order_id", "user_id"), on='order_id', how='left')
            .groupBy("user_Id", "order_id")
            .agg(F.collect_list(F.col("reordered")).alias("reordered_array"))
            .withColumn("doesnt_contains_reordered", F.when(F.array_contains("reordered_array", 1), 0).otherwise(1))
            .select("order_id", "doesnt_contains_reordered")
        )

        result_df = (
            self.prior_orders_df
            .join(df_with_num_of_reord, on="user_id", how='left')
            .join(df_with_time_since_prev_ord, on="order_id", how="left")
            .join(df_with_does_usr_asian_gluten_orga_items_ord, on="order_id", how='left')
            .join(df_with_fets_of_ord_size, on="user_id", how='left')
            .join(df_with_freq_ord_that_hasnt_prev_purch_items, on="order_id", how='left')
        )
        long_cols = [field.name for field in result_df.schema.fields if isinstance(field.dataType, LongType)]
        columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}
        result_df = result_df.withColumns(columns_to_cast)
        return result_df
        
    def generate_product_related_features(self):
                
        # How often the item has been purchased
        df_with_freq_purch = (
            self.prior_product_orders.select("product_id", "order_id")
            .groupBy("product_id")
            .agg(F.count(F.col("order_id")).alias("product_count"))
        )
        
        # Position of product
        df_with_avg_position_of_prod = (
            self.prior_product_orders.select("product_id", "add_to_cart_order")
            .groupBy("product_id")
            .agg(F.mean(F.col("add_to_cart_order")).alias("product_mean_of_position"))
        )
        
        # How many users buy it as a "one-shot" item
        df_with_freq_one_shot_ord_prods = (
            self.prior_product_orders.select("order_id", "product_id")
            .groupBy("order_id")
            .agg(F.collect_list("product_id").alias("list_of_products"))
            .withColumn("is_one_shot_order", F.when(F.size(F.col("list_of_products")) == 1, 1).otherwise(0))
            .withColumn("product_id", F.explode(F.col("list_of_products")))
            .join(self.prior_orders_df.select("user_id", "order_id"), on="order_id", how='left')
            .groupBy("product_id", "user_id")
            .agg(F.collect_list(F.col("is_one_shot_order")).alias("is_one_shot_order_list"))
            .withColumn("has_user_purchased_one_shot", F.when(F.array_contains("is_one_shot_order_list", 1), 1).otherwise(0))
            .groupBy("product_id")
            .agg(F.sum(F.col("has_user_purchased_one_shot")).alias("number_of_user_purchased_item"))
        )
        
        # Statistics on the number of items that co-occur with this item
        df_with_freq_co_ocrd = (
            self.prior_product_orders
            .select("product_id", "order_id")
            .alias("df1")
            .join(self.prior_product_orders.select("product_id", "order_id").withColumnRenamed("product_id", "product_id_1").alias("df2"),
                  (F.col("df1.order_id") == F.col("df2.order_id")) & (F.col("df1.product_id") != F.col("df2.product_id_1")),
                  "left")
            .groupBy("df1.product_id")
            .agg(F.count(F.col("df2.product_id_1")).alias("number_of_product_co_occurred"))
        )
        
        # Average number of items that co-occur with this item in a single order
        df_with_avg_num_item_co_ocrd_in_ord = (
            self.prior_product_orders.select("product_id", "order_id").alias("ppo1")
            .join(self.prior_product_orders.select("product_id", "order_id").alias("ppo2"),
                  (F.col("ppo1.order_id") == F.col("ppo2.order_id")) & (F.col("ppo1.product_id") != F.col("ppo2.product_id")),
                  how='left')
            .groupBy("ppo1.product_id", "ppo1.order_id")
            .agg(F.count(F.col("ppo2.product_id")).alias("count_of_co_ocuured_product_per_order"))
            .groupBy("ppo1.product_id")
            .agg(F.mean(F.col("count_of_co_ocuured_product_per_order")).alias("mean_of_co_ocuured_product_per_order"),
                 F.min(F.col("count_of_co_ocuured_product_per_order")).alias("min_of_co_ocuured_product_per_order"),
                 F.max(F.col("count_of_co_ocuured_product_per_order")).alias("max_of_co_ocuured_product_per_order"))
        )
        
        # Stats on the order streak
        df_with_flag = (
            self.prior_product_orders.select("product_id", "order_id")
            .join(self.prior_orders_df.select("user_id", "order_number", "order_id"), how='left', on='order_id')
            .withColumn("next_order_number", F.lead(F.col("order_number"), 1).over(Window.partitionBy("user_id", "product_id").orderBy("order_number")))
            .withColumn("is_streak_continued_flag", F.when(F.col("next_order_number") - F.col("order_number") == 1, 1).otherwise(0))
        )
        
        w1 = Window.partitionBy("user_id", "product_id").orderBy("order_number")
        w2 = Window.partitionBy("user_id", "product_id", "is_streak_continued_flag").orderBy("order_number")
        
        df_with_streak_length = (
            df_with_flag.withColumn("grp", F.row_number().over(w1) - F.row_number().over(w2))
            .groupBy("user_id", "product_id", "grp")
            .agg(F.count("order_number").alias("length_of_streaks"))
        )
        
        df_with_stats_of_streaks = (
            df_with_streak_length.select("product_id", "length_of_streaks", "grp")
            .groupBy("product_id")
            .agg(F.count('grp').alias("Total_streak_of_this_product"),
                 F.mean("length_of_streaks").alias("mean_of_streaks_of_this_product"),
                 F.min("length_of_streaks").alias("min_of_streaks_of_this_product"),
                 F.max("length_of_streaks").alias("max_of_streaks_of_this_product"))
        )
        
        # Probability of being reordered within N orders
        df_with_prob_greater_5 = (
            df_with_streak_length.withColumn("is_streak_length_greater_than_5", F.when(F.col("length_of_streaks") >= 5, 1).otherwise(0))
            .groupBy("product_id")
            .agg(F.count("length_of_streaks").alias("total_streaks"),
                 F.sum("is_streak_length_greater_than_5").alias("total_streaks_greater_than_5"))
            .withColumn("prob_of_reordered_5", F.col("total_streaks_greater_than_5") / F.col("total_streaks"))
            .select("product_id", "prob_of_reordered_5")
        )
        
        df_with_prob_greater_2 = (
            df_with_streak_length.withColumn("is_streak_length_greater_than_2", F.when(F.col("length_of_streaks") >= 2, 1).otherwise(0))
            .groupBy("product_id")
            .agg(F.count("length_of_streaks").alias("total_streaks"),
                 F.sum("is_streak_length_greater_than_2").alias("total_streaks_greater_than_2"))
            .withColumn("prob_of_reordered_2", F.col("total_streaks_greater_than_2") / F.col("total_streaks"))
            .select("product_id", "prob_of_reordered_2")
        )
        
        df_with_prob_greater_3 = (
            df_with_streak_length.withColumn("is_streak_length_greater_than_3", F.when(F.col("length_of_streaks") >= 3, 1).otherwise(0))
            .groupBy("product_id")
            .agg(F.count("length_of_streaks").alias("total_streaks"),
                 F.sum("is_streak_length_greater_than_3").alias("total_streaks_greater_than_3"))
            .withColumn("prob_of_reordered_3", F.col("total_streaks_greater_than_3") / F.col("total_streaks"))
            .select("product_id", "prob_of_reordered_3")
        )
        
        # Distribution of the day of week it is ordered
        pivoted_prior_orders_df = (
            self.prior_orders_df.select("order_id", "order_dow")
            .groupBy("order_id")
            .pivot("order_dow")
            .agg(F.lit(1)).na.fill(0)
        )
        
        df_with_count_of_dow_p_prod = (
            self.prior_product_orders.select("order_id", "product_id")
            .join(pivoted_prior_orders_df, on="order_id", how='left')
            .groupBy("product_id")
            .agg(F.sum("0").alias("distrib_count_of_dow_0_p_prod"),
                 F.sum("1").alias("distrib_count_of_dow_1_p_prod"),
                 F.sum("2").alias("distrib_count_of_dow_2_p_prod"),
                 F.sum("3").alias("distrib_count_of_dow_3_p_prod"),
                 F.sum("4").alias("distrib_count_of_dow_4_p_prod"),
                 F.sum("5").alias("distrib_count_of_dow_5_p_prod"),
                 F.sum("6").alias("distrib_count_of_dow_6_p_prod"))
        )
        
        # Probability it is reordered after the first order
        total_orders = self.prior_orders_df.select("order_id").distinct().count()
        
        df_with_prob_reord = (
            self.prior_orders_df.select("order_id", "user_id")
            .join(self.prior_product_orders.select("product_id", "order_id"), on="order_id", how='left')
            .groupBy("product_id", "user_id")
            .agg(F.count("order_id").alias("order_count"))
            .groupBy("product_id")
            .agg(((F.sum("order_count") / total_orders).alias("prob_of_being_reordered")))
        )
        
        result_product_df = (
            df_with_avg_position_of_prod
            .join(df_with_freq_one_shot_ord_prods, on="product_id", how='left')
            .join(df_with_freq_co_ocrd, on="product_id", how='left')
            .join(df_with_avg_num_item_co_ocrd_in_ord, df_with_avg_num_item_co_ocrd_in_ord["ppo1.product_id"] == self.prior_product_orders["product_id"], how="left")
            .join(df_with_stats_of_streaks, on="product_id", how='left')
            .join(df_with_prob_greater_5, on="product_id", how="left")
            .join(df_with_prob_greater_3, on="product_id", how="left")
            .join(df_with_prob_greater_2, on="product_id", how="left")
            .join(df_with_count_of_dow_p_prod, on="product_id", how="left")
            .join(df_with_prob_reord, on="product_id", how="left")
            .join(self.products_df.drop("product_name"), on="product_id", how="left")
        )

        long_cols = [field.name for field in result_product_df.schema.fields if isinstance(field.dataType, LongType)]
        columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}
        result_product_df = result_product_df.withColumns(columns_to_cast)
        return result_product_df
        
    def generate_user_product_related_features(self):
        
        # Number of orders in which the user purchases the item
        df_with_num_of_order_p_product = (
            self.prior_product_orders.select("order_id", "product_id")
            .join(self.prior_orders_df.select("order_id", "user_id"), how='left', on='order_id')
            .groupBy("user_id", "product_id")
            .agg(F.count("order_id").alias("num_of_ord_purch_p_prod"))
        )
        
        # Position in the cart
        df_with_position_cart_p_usr_p_prod = (
            self.prior_product_orders.select("product_id", "add_to_cart_order", "order_id")
            .join(self.prior_orders_df.select("user_id", "order_id"), how='left', on='order_id')
            .groupBy("user_id", "product_id")
            .agg(F.mean(F.col("add_to_cart_order")).alias("prod_mean_of_position_p_user"))
        )
        
        # Co-occurrence statistics
        df_with_co_ocrd_stats_p_user_p_prod = (
            self.prior_product_orders.select("product_id", "order_id").alias("df1")
            .join(self.prior_orders_df.select("user_id", "order_id"), on='order_id', how='left')
            .join(self.prior_product_orders.select("product_id", "order_id").withColumnRenamed("product_id", "product_id_1").alias("df2"),
                  (F.col("df1.order_id") == F.col("df2.order_id")) & (F.col("df1.product_id") != F.col("df2.product_id_1")),
                  "left")
            .groupBy("user_id", "df1.product_id")
            .agg(F.count(F.col("df2.product_id_1")).alias("num_of_prod_co_ocrd_p_usr_p_prod"))
        )

        result_usr_prod_df = (
            self.prior_product_orders
            .withColumnRenamed("product_id", "product_id_p")
            .alias("ppo")
            .join(self.prior_orders_df.select("user_id", "order_id").withColumnRenamed("user_id", "user_id_p").alias("pod"), on="order_id", how="left")
            .join(df_with_num_of_order_p_product,
                  (F.col("pod.user_id_p") == df_with_num_of_order_p_product['user_id']) &
                  (F.col("ppo.product_id_p") == df_with_num_of_order_p_product['product_id']), how="left").drop("user_id", "product_id")
            .join(df_with_position_cart_p_usr_p_prod,
                  (df_with_position_cart_p_usr_p_prod["user_id"] == F.col("pod.user_id_p")) &
                  (df_with_position_cart_p_usr_p_prod["product_id"] == F.col("ppo.product_id_p")), how="left").drop("user_id", "product_id")
            .join(df_with_co_ocrd_stats_p_user_p_prod,
                  (df_with_co_ocrd_stats_p_user_p_prod["user_id"] == F.col("pod.user_id_p")) &
                  (df_with_co_ocrd_stats_p_user_p_prod["product_id"] == F.col("ppo.product_id_p")), how="left").drop("user_id", "product_id")
        )
        long_cols = [field.name for field in result_usr_prod_df.schema.fields if isinstance(field.dataType, LongType)]
        columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}
        result_usr_prod_df = result_usr_prod_df.withColumns(columns_to_cast)
        return result_usr_prod_df

    def generate_time_related_features(self):
        # Counts by day of the week
        result_df = self.generate_user_related_features()
        
        df_with_count_of_dow = (
            self.prior_orders_df.select("order_id", "order_dow")
            .groupBy("order_dow")
            .agg(F.count("order_id").alias("total_ord_count_p_dow"))
        )
        
        # Counts by hour of the day
        df_with_count_of_ohod = (
            self.prior_orders_df.select("order_id", "order_hour_of_day")
            .groupBy("order_hour_of_day")
            .agg(F.count("order_id").alias("total_ord_count_p_ohod"))
        )
        result_time_df = (
            result_df.join(df_with_count_of_dow, on="order_dow", how="left")
            .join(df_with_count_of_ohod, on="order_hour_of_day", how="left")
        )
        
        
        long_cols = [field.name for field in result_df_with_time_df.schema.fields if isinstance(field.dataType, LongType)]
        columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}
        result_df_with_time_df = result_df_with_time_df.withColumns(columns_to_cast)
        return result_df_with_time_df


    def generate_all_types_of_features(self):
        
        result_usr_prod_df = self.generate_user_product_related_features()
        result_df_with_time_df = self.generate_time_related_features()
        result_product_df = self.generate_product_related_features()
        
        final_prior_ord_train_df = (
            result_usr_prod_df
            .join(result_df_with_time_df.drop("user_id"), on="order_id", how="left")
            .join(result_product_df.drop("user_id"), (F.col("product_id_p") == result_product_df['ppo1.product_id']), how="left")
            .drop("product_id")
            .withColumnsRenamed({"user_id_p":"user_id","product_id_p":"product_id"})
        )
        
        return final_prior_ord_train_df

        
def generate_test_set_features(user_stats_df,prods_stats_df,user_prod_stats_df,time_related_stats,test_set):
        
    df_list = [user_stats_df,prods_stats_df,user_prod_stats_df]
    
    for i in df_list:
        
        if "user_id" not in i.columns:
            raise NameError(f"'user_id' is missing in {i}")
        
        elif "product_id" not in i.columns:
            raise NameError(f"'product_id' is missing in {i}")
        
    if "user_id"  not in test_set.columns and "product_id" not in test_set.columns:
        raise NameError("'user_id' and 'product_id' both are missing in test_set")
        
    elif "user_id" not in test_set.columns:
        raise NameError("'user_id' not found in test_set")
        
    elif "product_id" not in test_set.columns:
        raise NameError("'product_id' not found in test_set")
        
    else:
        mean_dow_value = time_related_stats.groupBy("order_dow").agg(
            F.mean("total_ord_count_p_dow")
        ).collect()[0][0]
        
        mean_ohod_value = time_related_stats.groupBy("order_hour_of_day").agg(
            F.mean("total_ord_count_p_ohod")
        ).collect()[0][0]
        
        result_test_df = (
              
            test_set.join(
                user_stats_df, on = 'user_id', how = 'inner'
            )
            .join(
                prods_stats_df, on = 'product_id', how = 'inner'
            )
            .join(
                user-prod_stats_df, on = ['user_id','product_id'], how = 'inner'
            )
            .withColumns({
                          "time_mean_dow_count":F.lit(mean_dow_value),
                          "time_mean_ohod_count":F.lit(mean_ohod_value)
            })
        )
        
    return result_test_df