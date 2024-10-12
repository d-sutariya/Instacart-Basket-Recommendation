# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:16:10.630401Z","iopub.execute_input":"2024-10-12T13:16:10.630834Z","iopub.status.idle":"2024-10-12T13:17:05.822312Z","shell.execute_reply.started":"2024-10-12T13:16:10.630791Z","shell.execute_reply":"2024-10-12T13:17:05.820994Z"},"jupyter":{"outputs_hidden":false}}
!pip install pyspark

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:17:05.825145Z","iopub.execute_input":"2024-10-12T13:17:05.825863Z","iopub.status.idle":"2024-10-12T13:17:05.928960Z","shell.execute_reply.started":"2024-10-12T13:17:05.825804Z","shell.execute_reply":"2024-10-12T13:17:05.927709Z"},"jupyter":{"outputs_hidden":false}}
import time
import pyspark
import numpy as np
from pyspark.sql import SparkSession , Window
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, DoubleType

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:17:05.930273Z","iopub.execute_input":"2024-10-12T13:17:05.930811Z","iopub.status.idle":"2024-10-12T13:17:11.601798Z","shell.execute_reply.started":"2024-10-12T13:17:05.930773Z","shell.execute_reply":"2024-10-12T13:17:11.600415Z"},"jupyter":{"outputs_hidden":false}}
# Create a SparkSession with custom memory settings
spark = SparkSession.builder.appName("instamart_analysis") \
    .config("spark.driver.memory","25g") \
    .getOrCreate()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:17:11.607356Z","iopub.execute_input":"2024-10-12T13:17:11.607806Z","iopub.status.idle":"2024-10-12T13:17:11.613754Z","shell.execute_reply.started":"2024-10-12T13:17:11.607760Z","shell.execute_reply":"2024-10-12T13:17:11.612320Z"},"jupyter":{"outputs_hidden":false}}
def show_time(start):
    return time.time() - start

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:17:11.615421Z","iopub.execute_input":"2024-10-12T13:17:11.615865Z","iopub.status.idle":"2024-10-12T13:18:01.095633Z","shell.execute_reply.started":"2024-10-12T13:17:11.615821Z","shell.execute_reply":"2024-10-12T13:18:01.094412Z"},"jupyter":{"outputs_hidden":false}}
departments_df = spark.read.options(header=True,inferSchema=True).csv("/kaggle/input/instacart-market-basket-analysis/departments.csv")
products_df = spark.read.options(header=True,inferSchema=True).csv("/kaggle/input/instacart-market-basket-analysis/products.csv")
prior_product_orders = spark.read.options(header=True,inferSchema=True).csv("/kaggle/input/instacart-market-basket-analysis/order_products__prior.csv").repartition(12)
train_product_orders = spark.read.options(header=True,inferSchema=True).csv("/kaggle/input/instacart-market-basket-analysis/order_products__train.csv").repartition(8)
orders_df = spark.read.options(header=True,inferSchema=True).csv("/kaggle/input/instacart-market-basket-analysis/orders.csv").repartition(8)
aisels_df = spark.read.options(header=True,inferSchema=True).csv("/kaggle/input/instacart-market-basket-analysis/aisles.csv")

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:01.099209Z","iopub.execute_input":"2024-10-12T13:18:01.099678Z","iopub.status.idle":"2024-10-12T13:18:05.065481Z","shell.execute_reply.started":"2024-10-12T13:18:01.099635Z","shell.execute_reply":"2024-10-12T13:18:05.064400Z"},"jupyter":{"outputs_hidden":false}}
orders_df.filter(F.col('eval_set') == 'test').count()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:05.111077Z","iopub.execute_input":"2024-10-12T13:18:05.111620Z","iopub.status.idle":"2024-10-12T13:18:05.165284Z","shell.execute_reply.started":"2024-10-12T13:18:05.111562Z","shell.execute_reply":"2024-10-12T13:18:05.164068Z"},"jupyter":{"outputs_hidden":false}}
train_orders_df = orders_df.filter(orders_df["eval_set"] =='train').drop("eval_set")
prior_orders_df = orders_df.filter(orders_df["eval_set"] == 'prior').drop("eval_set")

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:05.167317Z","iopub.execute_input":"2024-10-12T13:18:05.167865Z","iopub.status.idle":"2024-10-12T13:18:05.305211Z","shell.execute_reply.started":"2024-10-12T13:18:05.167800Z","shell.execute_reply":"2024-10-12T13:18:05.303981Z"},"jupyter":{"outputs_hidden":false}}
# how often user has reorderd
class FeatureGenerator:
    
    def  __init__(self,train_set,test_set=None):
       self.train_set = train_set
       self.test_set = test_set

    def generate_user_related_features(self,test_set=None):
        
df_with_num_of_reord = (
    prior_product_orders.select("reordered","order_id").join(
        prior_orders_df.select("user_id","order_id"),how="left",on="order_id"
    ).select("user_id","reordered") 
     .groupBy("user_id").agg(
        F.count(F.col("reordered")).alias("frequency of reorder")
    )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:05.306580Z","iopub.execute_input":"2024-10-12T13:18:05.309201Z","iopub.status.idle":"2024-10-12T13:18:05.475460Z","shell.execute_reply.started":"2024-10-12T13:18:05.309136Z","shell.execute_reply":"2024-10-12T13:18:05.474211Z"},"jupyter":{"outputs_hidden":false}}
# time since privious order

df_with_time_since_prev_ord = (
    prior_orders_df.select("user_id","days_since_prior_order","order_hour_of_day","order_number","order_id") 
                .withColumn("privious_order_hour",
                            F.lag("order_hour_of_day",1) 
                            .over(Window.partitionBy("user_id").orderBy("order_number"))) 
                .withColumn("time_since_Last_order",
                            F.col("days_since_prior_order") * 24 + 
                            F.col("order_hour_of_day") - 
                            F.col("privious_order_hour") 
                           ) 
                .select("order_id","time_since_last_order")
)


#time of the day user visits

df_with_time_of_day_usr_visits = (
    prior_orders_df.select("user_id" , "order_hour_of_day","order_id") 
                .groupBy("user_id","order_hour_of_day") 
                .agg(F.count("order_id").alias("frequency")) 
                .groupBy("user_id") 
                .agg(F.max("frequency").alias("maximum_frquency"))
)

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2024-10-12T13:18:05.569308Z","iopub.execute_input":"2024-10-12T13:18:05.569759Z","iopub.status.idle":"2024-10-12T13:18:05.952811Z","shell.execute_reply.started":"2024-10-12T13:18:05.569707Z","shell.execute_reply":"2024-10-12T13:18:05.951562Z"},"jupyter":{"outputs_hidden":false}}
# does the user have ordered asian , gluten free, or organic item 

df_with_does_usr_asian_gluten_orga_items_ord = (
    prior_product_orders.select("order_id","product_id") 
            .join(products_df.select("product_id","product_name"), on="product_id", how='left') 
            .join(prior_orders_df.select("user_id","order_id"), on="order_id", how='left') 
            .groupBy("user_id", "order_id") 
            .agg(F.collect_list("product_name").alias("list_of_products")) 
            .withColumn("normalized_list", F.expr("transform(list_of_products, x -> lower(x))")) 
            .withColumn("contains_or_not", 
                F.expr("exists(normalized_list,x -> x like '%organic%')")
              | F.expr("exists(normalized_list, x -> x like '%asian%')")
              | F.expr("exists(normalized_list, x-> x like '%gluten free%')")
            ) 
            .select("order_id","contains_or_not")
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:05.954132Z","iopub.execute_input":"2024-10-12T13:18:05.954584Z","iopub.status.idle":"2024-10-12T13:18:06.074720Z","shell.execute_reply.started":"2024-10-12T13:18:05.954535Z","shell.execute_reply":"2024-10-12T13:18:06.073351Z"},"jupyter":{"outputs_hidden":false}}
# feature based on order size 

df_with_fets_of_ord_size = (
    prior_product_orders.select("product_id","order_id") 
                    .join(prior_orders_df.select("user_id","order_id") , on="order_id", how="left") 
                    .groupBy("user_id",'order_id') 
                    .agg(
                            F.count(F.col("product_id")).alias("count_of_product")
                        ) 
                    .groupBy("user_id") 
                    .agg(
                            F.max(F.col("count_of_product")).alias("max_count_of_products"),
                            F.min(F.col("count_of_product")).alias("min_count_of_products"),
                            F.mean(F.col("count_of_product")).alias("mean_count_of_products")
                        ) 
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:06.076284Z","iopub.execute_input":"2024-10-12T13:18:06.076722Z","iopub.status.idle":"2024-10-12T13:18:06.193317Z","shell.execute_reply.started":"2024-10-12T13:18:06.076675Z","shell.execute_reply":"2024-10-12T13:18:06.191905Z"},"jupyter":{"outputs_hidden":false}}
# How many of the user’s orders contained no previously purchased items

df_with_freq_ord_that_hasnt_prev_purch_items = (
    prior_product_orders.select("order_id","reordered") 
                    .join(prior_orders_df.select("order_id","user_id") , on = 'order_id' , how = 'left') 
                    .groupBy("user_Id","order_id") 
                    .agg(
                            F.collect_list(F.col("reordered")).alias("reordered_array")
                        ) 
                    .withColumn("doesnt_contains_reordered" ,
                            F.when(F.array_contains("reordered_array",1),0).otherwise(1)
                        ) 
                    .select("order_id","doesnt_contains_reordered")
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:06.194927Z","iopub.execute_input":"2024-10-12T13:18:06.195399Z","iopub.status.idle":"2024-10-12T13:18:06.234191Z","shell.execute_reply.started":"2024-10-12T13:18:06.195346Z","shell.execute_reply":"2024-10-12T13:18:06.232782Z"},"jupyter":{"outputs_hidden":false}}
# how often the item has been purchaced 

df_with_freq_purch= (
    prior_product_orders.select("product_id","order_id") 
                     .groupBy("product_id") 
                     .agg(
                             F.count(F.col("order_id")).alias("product_count")
                        ) 
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:06.235815Z","iopub.execute_input":"2024-10-12T13:18:06.236276Z","iopub.status.idle":"2024-10-12T13:18:06.273648Z","shell.execute_reply.started":"2024-10-12T13:18:06.236228Z","shell.execute_reply":"2024-10-12T13:18:06.272482Z"},"jupyter":{"outputs_hidden":false}}
# position of product 

df_with_avg_position_of_prod = (
    prior_product_orders.select("product_id","add_to_cart_order") 
                    .groupBy("product_id") 
                    .agg(
                            F.mean(F.col("add_to_cart_order")).alias("product_mean_of_position")
                        ) 
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:06.274874Z","iopub.execute_input":"2024-10-12T13:18:06.275311Z","iopub.status.idle":"2024-10-12T13:18:06.470033Z","shell.execute_reply.started":"2024-10-12T13:18:06.275261Z","shell.execute_reply":"2024-10-12T13:18:06.468875Z"},"jupyter":{"outputs_hidden":false}}
# How many users buy it as "one shot" item

df_with_freq_one_shot_ord_prods = (
    prior_product_orders.select("order_id","product_id") 
                    .groupBy("order_id") 
                    .agg(F.collect_list("product_id").alias("list_of_products")) 
                    .withColumn("is_one_shot_order",
                                   F.when(F.size(F.col("list_of_products")) == 1,1).otherwise(0)
                               ) 
                    .withColumn("product_id",F.explode(F.col("list_of_products"))) 
                    .join(prior_orders_df.select("user_id","order_id"),on="order_id",how='left') 
                    .groupBy("product_id","user_id") 
                    .agg(F.collect_list(F.col("is_one_shot_order")).alias("is_one_shot_order_list")) 
                    .withColumn("has_user_purchased_one_shot",F.when(F.array_contains("is_one_shot_order_list",1),1).otherwise(0)) 
                    .groupBy("product_id") 
                    .agg(
                            F.sum(F.col("has_user_purchased_one_shot")).alias("number_of_user_purchased_item")
                        ) 
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:06.471344Z","iopub.execute_input":"2024-10-12T13:18:06.471757Z","iopub.status.idle":"2024-10-12T13:18:06.724250Z","shell.execute_reply.started":"2024-10-12T13:18:06.471710Z","shell.execute_reply":"2024-10-12T13:18:06.722944Z"},"jupyter":{"outputs_hidden":false}}
# Stats on the number of items that co-occur with this item

# 1. number of time that a item has co occured.

# Perform a self-join on prior_product_orders
df_with_freq_co_ocrd = (
    prior_product_orders
    .select("product_id", "order_id")
    .alias("df1")
    .join(
        prior_product_orders.select("product_id", "order_id")
        .withColumnRenamed("product_id", "product_id_1")
        .alias("df2"),
        (F.col("df1.order_id") == F.col("df2.order_id")) & (F.col("df1.product_id") != F.col("df2.product_id_1")),
        "left"
    )
    .groupBy("df1.product_id")
    .agg(F.count(F.col("df2.product_id_1")).alias("number_of_product_co_occurred"))
)

# 2 average number of items that is co ocuured with this item in single order

df_with_avg_num_item_co_ocrd_in_ord = (
                prior_product_orders.select("product_id","order_id").alias("ppo1") 
                .join(
                    prior_product_orders.select("product_id","order_id")
                    .alias("ppo2"),
                    (F.col("ppo1.order_id") == F.col("ppo2.order_id")) & 
                    (F.col("ppo1.product_id") != F.col("ppo2.product_id")),
                    how='left'
                ) 
                .groupBy("ppo1.product_id","ppo1.order_id")
                .agg(F.count(F.col("ppo2.product_id")).alias("count_of_co_ocuured_product_per_order"))
                .groupBy("ppo1.product_id")
                .agg(
                    F.mean(F.col("count_of_co_ocuured_product_per_order")).alias("mean_of_co_ocuured_product_per_order"),
                    F.min(F.col("count_of_co_ocuured_product_per_order")).alias("min_of_co_ocuured_product_per_order"),
                    F.max(F.col("count_of_co_ocuured_product_per_order")).alias("max_of_co_ocuured_product_per_order"),

                )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:06.725594Z","iopub.execute_input":"2024-10-12T13:18:06.726042Z","iopub.status.idle":"2024-10-12T13:18:06.969590Z","shell.execute_reply.started":"2024-10-12T13:18:06.725987Z","shell.execute_reply":"2024-10-12T13:18:06.968558Z"},"jupyter":{"outputs_hidden":false}}
# Stats on the order streak

# 1. let's add the flag whether streak is continued or not

df_with_flag= (

    prior_product_orders.select("product_id","order_id")
                        .join(
                                prior_orders_df.select("user_id","order_number","order_id"),
                                how ='left',
                                on = 'order_id' 
                            )
                        .withColumn("next_order_number",
                            F.lead(F.col("order_number"),1).over(Window.partitionBy("user_id","product_id").orderBy("order_number"))
                        )
                        .withColumn("is_streak_continued_flag",
                               F.when(F.col("next_order_number") - F.col("order_number") == 1,1)
                                    .otherwise(0)
                            )
)
# 2. let's assign an unique id to each streak of a perticular user and product.

w1 = Window.partitionBy("user_id","product_id").orderBy("order_number")
w2 = Window.partitionBy("user_id","product_id","is_streak_continued_flag").orderBy("order_number")

# by using the above window we can create unique id for streak named grp then can find streak leangth.
df_with_streak_length = (
    df_with_flag.withColumn("grp",F.row_number().over(w1) - F.row_number().over(w2))
                .groupBy("user_id","product_id","grp")
                .agg(
                    F.count("order_number").alias("length_of_streaks")
                )
)

# finally , summarize it over each prodcut rather than per user per product.
df_with_stats_of_streaks = (
    df_with_streak_length.select("product_id","length_of_streaks","grp")
                         .groupBy("product_id")
                         .agg(
                             F.count('grp').alias("Total_streak_of_this_product"),
                             F.mean("length_of_streaks").alias("mean_of_streaks_of_this_product"),
                             F.min("length_of_streaks").alias("max_of_streaks_of_this_product"),
                             F.max("length_of_streaks").alias("min_of_streaks_of_this_product")
                         
                         )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:06.970746Z","iopub.execute_input":"2024-10-12T13:18:06.972275Z","iopub.status.idle":"2024-10-12T13:18:07.056123Z","shell.execute_reply.started":"2024-10-12T13:18:06.972218Z","shell.execute_reply":"2024-10-12T13:18:07.054944Z"},"jupyter":{"outputs_hidden":false}}
# Probability of being reordered within N orders

# we have already counted the lenght of the streaks so if it is >= 5 then it will be added in probability.

df_with_prob_greater_5 = (
    df_with_streak_length.withColumn("is_streak_length_greater_than_5",
                                        F.when(F.col("length_of_streaks") >= 5,1).otherwise(0) 
                                    )
                         .groupBy("product_id")
                         .agg(
                             F.count("length_of_streaks").alias("total_streaks"),
                             F.sum("is_streak_length_greater_than_5").alias("total_streaks_greater_than_5")
                         )
                         .withColumn("prob_of_reordered_5",
                             ( F.col("total_streaks_greater_than_5") / F.col("total_streaks"))
                         )
                         .select("product_id","prob_of_reordered_5")
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:07.058336Z","iopub.execute_input":"2024-10-12T13:18:07.058977Z","iopub.status.idle":"2024-10-12T13:18:07.139031Z","shell.execute_reply.started":"2024-10-12T13:18:07.058904Z","shell.execute_reply":"2024-10-12T13:18:07.138195Z"},"jupyter":{"outputs_hidden":false}}
# we have already counted the lenght of the streaks so if it is >= 2 then it will be added in probability.

df_with_prob_greater_2 = (
    df_with_streak_length.withColumn("is_streak_length_greater_than_2",
                                        F.when(F.col("length_of_streaks") >= 2,1).otherwise(0) 
                                    )
                         .groupBy("product_id")
                         .agg(
                             F.count("length_of_streaks").alias("total_streaks"),
                             F.sum("is_streak_length_greater_than_2").alias("total_streaks_greater_than_2")
                         )
                         .withColumn("prob_of_reordered_2",
                             ( F.col("total_streaks_greater_than_2") / F.col("total_streaks"))
                         )
                         .select("product_id","prob_of_reordered_2")
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:07.139982Z","iopub.execute_input":"2024-10-12T13:18:07.140331Z","iopub.status.idle":"2024-10-12T13:18:07.235443Z","shell.execute_reply.started":"2024-10-12T13:18:07.140294Z","shell.execute_reply":"2024-10-12T13:18:07.234173Z"},"jupyter":{"outputs_hidden":false}}
# we have already counted the lenght of the streaks so if it is >= 3 then it will be added in probability.

df_with_prob_greater_3 = (
    df_with_streak_length.withColumn("is_streak_length_greater_than_3",
                                        F.when(F.col("length_of_streaks") >= 3,1).otherwise(0) 
                                    )
                         .groupBy("product_id")
                         .agg(
                             F.count("length_of_streaks").alias("total_streaks"),
                             F.sum("is_streak_length_greater_than_3").alias("total_streaks_greater_than_3")
                         )
                         .withColumn("prob_of_reordered_3",
                             ( F.col("total_streaks_greater_than_3") / F.col("total_streaks"))
                         )
                         .select("product_id","prob_of_reordered_3")
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:07.241267Z","iopub.execute_input":"2024-10-12T13:18:07.241781Z","iopub.status.idle":"2024-10-12T13:18:14.593574Z","shell.execute_reply.started":"2024-10-12T13:18:07.241727Z","shell.execute_reply":"2024-10-12T13:18:14.592212Z"},"jupyter":{"outputs_hidden":false}}
# Distribution of the day of week it is ordered

pivoted_prior_orders_df = (
    prior_orders_df.select("order_id","order_dow")
                    .groupBy("order_id")
                    .pivot("order_dow")
                    .agg(F.lit(1)).na.fill(0)
)
            
df_with_count_of_dow_p_prod = (
    prior_product_orders.select("order_id","product_id")
                            .join(
                                pivoted_prior_orders_df , on = "order_id",how='left'
                            )
                            .groupBy("product_id")
                            .agg(
                                F.sum("0").alias("distrib_count_of_dow_0_p_prod"),
                                F.sum("1").alias("distrib_count_of_dow_1_p_prod"),
                                F.sum("2").alias("distrib_count_of_dow_2_p_prod"),
                                F.sum("3").alias("distrib_count_of_dow_3_p_prod"),
                                F.sum("4").alias("distrib_count_of_dow_4_p_prod"),
                                F.sum("5").alias("distrib_count_of_dow_5_p_prod"),
                                F.sum("6").alias("distrib_count_of_dow_6_p_prod")
                            )

)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:14.597054Z","iopub.execute_input":"2024-10-12T13:18:14.599994Z","iopub.status.idle":"2024-10-12T13:18:24.030601Z","shell.execute_reply.started":"2024-10-12T13:18:14.599925Z","shell.execute_reply":"2024-10-12T13:18:24.028629Z"},"jupyter":{"outputs_hidden":false}}
#  Probability it is reordered after the first order
total_orders = prior_orders_df.select("order_id").distinct().count()

df_with_prob_reord = (
    prior_orders_df.select("order_id","user_id")
                    .join(prior_product_orders.select("product_id","order_id"),on="order_id",how='left')
                    .groupBy("product_id","user_id")
                    .agg(
                        F.count("order_id").alias("order_count")
                    )
                    .groupBy("product_id")
                    .agg(
                        ( 
                            (F.sum("order_count") / total_orders).alias("prob_of_being_reordered") 
                        )
                    )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.032021Z","iopub.execute_input":"2024-10-12T13:18:24.032514Z","iopub.status.idle":"2024-10-12T13:18:24.138331Z","shell.execute_reply.started":"2024-10-12T13:18:24.032460Z","shell.execute_reply":"2024-10-12T13:18:24.136243Z"},"jupyter":{"outputs_hidden":false}}
# Number of orders in which the user purchases the item

df_with_num_of_order_p_product = (
    
    prior_product_orders.select("order_id","product_id")
                        .join(
                            prior_orders_df.select("order_id","user_id")
                            , how = 'left' , on = 'order_id'
                        )
                        .groupBy("user_id","product_id")
                        .agg(
                            F.count("order_id").alias("num_of_ord_purch_p_prod")
                        )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.146846Z","iopub.execute_input":"2024-10-12T13:18:24.147412Z","iopub.status.idle":"2024-10-12T13:18:24.156749Z","shell.execute_reply.started":"2024-10-12T13:18:24.147357Z","shell.execute_reply":"2024-10-12T13:18:24.155433Z"},"jupyter":{"outputs_hidden":false}}
# # Days since the user last purchased the item

# w1 = Window.partitionBy("user_id","product_id").orderBy("order_number")
# df_with_next_order_p_prod = (
#     prior_product_orders.select("product_id","order_id")
#                         .join(
#                             prior_orders_df.select("user_id","order_id","order_number","days_since_prior_order")
#                                             .groupBy("user_id")
#                                             .agg(
#                                                 F.collect_list("days_since_prior_order").alias("list_of_days_since_prior_ord"),
#                                                 F.collect_list("order_number").alias("list_of_order_number")
#                                             )
#                             ,
#                             how='left',on='order_id'
#                         )
#                         .withColumn("pre_order_number",
#                             F.lag(F.col("order_number")).over(w1)
                                
#                         ).na.fill({'pre_order_number':0})
#                         .sort("user_id","product_id","order_number")
#                         .show()
                        
# )

# w2 = Window.partitionBy("user_id").orderBy("order_number") 
#                         .rowsBetween(
#                             F.when(F.col("pre_order_number") == 0,Window.unboundedPreceding).otherwise(F.col("pre_order_number")) ,
#                             F.col("order_number")
#                         )

# df_with_days_since_last_ord_p_prod = (
#     df_with_next_order_p_prod.join(
#                                 prior_orders_df.select("user_id","days_since_prior_orders"),
#                                 how='left',on='user_id'
#                             ).sort("user_id","order")
#                             .groupBy("user_id")
#                             .withColum("sum_day_since_last_order",
#                                 F.sum("days_since_prior_order").over(w2)
#                             )
# )

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.158591Z","iopub.execute_input":"2024-10-12T13:18:24.159950Z","iopub.status.idle":"2024-10-12T13:18:24.239367Z","shell.execute_reply.started":"2024-10-12T13:18:24.159887Z","shell.execute_reply":"2024-10-12T13:18:24.237489Z"},"jupyter":{"outputs_hidden":false}}
# Position in the cart
df_with_position_cart_p_usr_p_prod = (
    prior_product_orders.select("product_id","add_to_cart_order","order_id") 
                    .join(
                        prior_orders_df.select("user_id","order_id"),
                        how = 'left' , on = 'order_id'
                    )
                    .groupBy("user_id","product_id") 
                    .agg(
                            F.mean(F.col("add_to_cart_order")).alias("prod_mean_of_position_p_user")
                        )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.241000Z","iopub.execute_input":"2024-10-12T13:18:24.241556Z","iopub.status.idle":"2024-10-12T13:18:24.358735Z","shell.execute_reply.started":"2024-10-12T13:18:24.241489Z","shell.execute_reply":"2024-10-12T13:18:24.357506Z"},"jupyter":{"outputs_hidden":false}}
# Co-occurrence statistics

df_with_co_ocrd_stats_p_user_p_prod = (
    prior_product_orders
    .select("product_id", "order_id")
    .alias("df1")
    .join(prior_orders_df.select("user_id","order_id"),
         on = 'order_id',how='left'
         )
    .join(
        prior_product_orders.select("product_id", "order_id")
        .withColumnRenamed("product_id", "product_id_1")
        .alias("df2"),
        (F.col("df1.order_id") == F.col("df2.order_id")) & (F.col("df1.product_id") != F.col("df2.product_id_1")),
        "left"
    )
    .groupBy("user_id","df1.product_id")
    .agg(
        F.count(F.col("df2.product_id_1")).alias("num_of_prod_co_ocrd_p_usr_p_prod"),
    )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.360041Z","iopub.execute_input":"2024-10-12T13:18:24.360518Z","iopub.status.idle":"2024-10-12T13:18:24.424495Z","shell.execute_reply.started":"2024-10-12T13:18:24.360465Z","shell.execute_reply":"2024-10-12T13:18:24.423046Z"},"jupyter":{"outputs_hidden":false}}
#Counts by day of wee

df_with_count_of_dow = (
        prior_orders_df.select("order_id","order_dow")
                        .groupBy("order_dow")
                        .agg(
                                F.count("order_id").alias("total_ord_count_p_dow")
                            )
)

#Counts by hour

df_with_count_of_ohod = (
        prior_orders_df.select("order_id","order_hour_of_day")
                        .groupBy("order_hour_of_day")
                        .agg(
                                F.count("order_id").alias("total_ord_count_p_ohod")
                            )
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.426052Z","iopub.execute_input":"2024-10-12T13:18:24.426568Z","iopub.status.idle":"2024-10-12T13:18:24.597067Z","shell.execute_reply.started":"2024-10-12T13:18:24.426510Z","shell.execute_reply":"2024-10-12T13:18:24.595940Z"},"jupyter":{"outputs_hidden":false}}
result_df = (
    prior_orders_df.join(
        df_with_num_of_reord , on = "user_id" ,how = 'left'
    )
    .join(
        df_with_time_since_prev_ord , on = "order_id" , how = "left"
    )
    .join(
        df_with_does_usr_asian_gluten_orga_items_ord , on = "order_id" , how = 'left'
    )
    .join(
        df_with_fets_of_ord_size , on = 'user_id' , how = 'left'
    )
    .join(
        df_with_freq_ord_that_hasnt_prev_purch_items , on = "order_id",how='left'
    )
)
result_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.598281Z","iopub.execute_input":"2024-10-12T13:18:24.600876Z","iopub.status.idle":"2024-10-12T13:18:24.697371Z","shell.execute_reply.started":"2024-10-12T13:18:24.600818Z","shell.execute_reply":"2024-10-12T13:18:24.696212Z"},"jupyter":{"outputs_hidden":false}}
long_cols = [field.name for field in result_df.schema.fields if isinstance(field.dataType, LongType)]

# Create a dictionary where aggregatethe key is the column name, and the value is the cast operation
columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}

result_df = result_df.withColumns(columns_to_cast)
result_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:24.703935Z","iopub.execute_input":"2024-10-12T13:18:24.705810Z","iopub.status.idle":"2024-10-12T13:18:25.177542Z","shell.execute_reply.started":"2024-10-12T13:18:24.705734Z","shell.execute_reply":"2024-10-12T13:18:25.176317Z"},"jupyter":{"outputs_hidden":false}}
result_product_df = (
    df_with_avg_position_of_prod
    .join(
        df_with_freq_one_shot_ord_prods , on = 'product_id' , how = 'left'
    )
    .join(
        df_with_freq_co_ocrd , on = "product_id" , how = 'left'
    )
    .join(
        df_with_avg_num_item_co_ocrd_in_ord , df_with_avg_num_item_co_ocrd_in_ord["ppo1.product_id"] == prior_product_orders["product_id"] , how ="left"
    )
    .join(
        df_with_stats_of_streaks , on = 'product_id' , how = 'left'
    )
    .join(
        df_with_prob_greater_5 , on = 'product_id' , how = "left"
    )
    .join(
        df_with_prob_greater_3 , on = 'product_id' , how = "left"
    )
    .join(
        df_with_prob_greater_2 , on = 'product_id' , how = "left"
    )
    .join(
        df_with_count_of_dow_p_prod , on = 'product_id', how = 'left'
    )
    .join(
        df_with_prob_reord , on = 'product_id' , how = 'left'
    )
    .join(
        products_df.drop("product_name") , on = 'product_id' , how = 'left'
    )
)
result_product_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.179166Z","iopub.execute_input":"2024-10-12T13:18:25.180856Z","iopub.status.idle":"2024-10-12T13:18:25.357525Z","shell.execute_reply.started":"2024-10-12T13:18:25.180793Z","shell.execute_reply":"2024-10-12T13:18:25.356304Z"},"jupyter":{"outputs_hidden":false}}
long_cols = [field.name for field in result_product_df.schema.fields if isinstance(field.dataType, LongType)]

# Create a dictionary where aggregatethe key is the column name, and the value is the cast operation
columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}

result_product_df = result_product_df.withColumns(columns_to_cast)
result_product_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.358878Z","iopub.execute_input":"2024-10-12T13:18:25.359380Z","iopub.status.idle":"2024-10-12T13:18:25.504840Z","shell.execute_reply.started":"2024-10-12T13:18:25.359326Z","shell.execute_reply":"2024-10-12T13:18:25.503654Z"},"jupyter":{"outputs_hidden":false}}
result_usr_prod_df = (
    prior_product_orders
    .withColumnRenamed("product_id","product_id_p")
    .alias("ppo")
    .join(
        prior_orders_df.select("user_id","order_id").withColumnRenamed("user_id","user_id_p").alias("pod") , on = 'order_id' , how = 'left'
    )
    .join(
        df_with_num_of_order_p_product ,
            (F.col("pod.user_id_p") == df_with_num_of_order_p_product['user_id']) &
            (F.col("ppo.product_id_p") == df_with_num_of_order_p_product['product_id']) 
        , how = 'left'
    ).drop("user_id","product_id")
    .join(
        df_with_position_cart_p_usr_p_prod ,
        (df_with_position_cart_p_usr_p_prod["user_id"] == F.col("pod.user_id_p")) &
        (df_with_position_cart_p_usr_p_prod["product_id"] == F.col("ppo.product_id_p"))
        ,how = 'left'
    ).drop("user_id","product_id")
    .join(
        df_with_co_ocrd_stats_p_user_p_prod ,
        (df_with_co_ocrd_stats_p_user_p_prod["user_id"] == F.col("pod.user_id_p") ) &
        (df_with_co_ocrd_stats_p_user_p_prod["product_id"] == F.col("ppo.product_id_p"))
        , how = 'left'
    ).drop("user_id","product_id")
)
result_usr_prod_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.506175Z","iopub.execute_input":"2024-10-12T13:18:25.506674Z","iopub.status.idle":"2024-10-12T13:18:25.552414Z","shell.execute_reply.started":"2024-10-12T13:18:25.506599Z","shell.execute_reply":"2024-10-12T13:18:25.551199Z"},"jupyter":{"outputs_hidden":false}}
long_cols = [field.name for field in result_usr_prod_df.schema.fields if isinstance(field.dataType, LongType)]

# Create a dictionary where aggregatethe key is the column name, and the value is the cast operation
columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}

result_usr_prod_df = result_usr_prod_df.withColumns(columns_to_cast)
result_usr_prod_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.553719Z","iopub.execute_input":"2024-10-12T13:18:25.554201Z","iopub.status.idle":"2024-10-12T13:18:25.623788Z","shell.execute_reply.started":"2024-10-12T13:18:25.554147Z","shell.execute_reply":"2024-10-12T13:18:25.622660Z"},"jupyter":{"outputs_hidden":false}}
result_df_with_time_df = (
    result_df.join(
        df_with_count_of_dow , on = "order_dow" , how = 'left'
    )
    .join(
       df_with_count_of_ohod , on = 'order_hour_of_day' , how = 'left'
    )
)
result_df_with_time_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.625572Z","iopub.execute_input":"2024-10-12T13:18:25.626130Z","iopub.status.idle":"2024-10-12T13:18:25.670330Z","shell.execute_reply.started":"2024-10-12T13:18:25.626048Z","shell.execute_reply":"2024-10-12T13:18:25.669035Z"},"jupyter":{"outputs_hidden":false}}
long_cols = [field.name for field in result_df_with_time_df.schema.fields if isinstance(field.dataType, LongType)]

# Create a dictionary where aggregatethe key is the column name, and the value is the cast operation
columns_to_cast = {col_name: F.col(col_name).cast(DoubleType()) for col_name in long_cols}

result_df_with_time_df = result_df_with_time_df.withColumns(columns_to_cast)
result_df_with_time_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.675800Z","iopub.execute_input":"2024-10-12T13:18:25.676782Z","iopub.status.idle":"2024-10-12T13:18:25.889416Z","shell.execute_reply.started":"2024-10-12T13:18:25.676722Z","shell.execute_reply":"2024-10-12T13:18:25.888100Z"},"jupyter":{"outputs_hidden":false}}
final_prior_ord_train_df = (
    result_usr_prod_df.join(
        result_df_with_time_df.drop("user_id"),
        on = "order_id",how='left'
    )
    .join(
        result_product_df.drop("user_id") ,
        (F.col("product_id_p") == result_product_df['ppo1.product_id'])
        , how = 'left'
    ).drop("product_id")
)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.890774Z","iopub.execute_input":"2024-10-12T13:18:25.891189Z","iopub.status.idle":"2024-10-12T13:18:25.909597Z","shell.execute_reply.started":"2024-10-12T13:18:25.891106Z","shell.execute_reply":"2024-10-12T13:18:25.908292Z"},"jupyter":{"outputs_hidden":false}}
final_prior_ord_train_df.printSchema()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-12T13:18:25.911790Z","iopub.execute_input":"2024-10-12T13:18:25.912775Z","iopub.status.idle":"2024-10-12T13:18:25.918699Z","shell.execute_reply.started":"2024-10-12T13:18:25.912714Z","shell.execute_reply":"2024-10-12T13:18:25.917215Z"},"jupyter":{"outputs_hidden":false}}
# final_prior_ord_train_df.coalesce(1).write.mode('overwrite').csv('/kaggle/working/final_prior_ord_train_df',header=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
