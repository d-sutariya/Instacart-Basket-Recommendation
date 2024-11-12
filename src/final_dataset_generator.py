import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from instacart_feature_transformation_script import FeatureGenerator, generate_test_set_features


def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("instamart_analysis") \
        .config("spark.driver.memory", "25g") \
        .getOrCreate()

    # Get file paths from environment variables
    orders_file_path = os.getenv("ORDERS_FILE_PATH")
    prior_product_orders_file_path = os.getenv("PRIOR_PRODUCT_ORDERS_FILE_PATH")
    products_file_path = os.getenv("PRODUCTS_FILE_PATH")
    train_product_orders_file_path = os.getenv("TRAIN_PRODUCT_ORDERS_FILE_PATH")

    if not all([orders_file_path, prior_product_orders_file_path, products_file_path, train_product_orders_file_path]):
        raise ValueError("Please set all the required file paths in environment variables.")

    # Load datasets
    orders_df = spark.read.csv(orders_file_path, header=True)
    prior_product_orders = spark.read.csv(prior_product_orders_file_path, header=True)
    products_df = spark.read.csv(products_file_path, header=True)
    train_product_orders = spark.read.csv(train_product_orders_file_path, header=True)

    # Type casting for prior product orders dataframe
    prior_product_orders = prior_product_orders.select(
        [F.col(col).cast("float").alias(col) for col in prior_product_orders.columns]
    )

    # Convert string column to float for orders dataframe
    orders_df = orders_df.select(
        [F.col(col).cast("float").alias(col) if col != 'eval_set' else F.col(col).alias(col) for col in orders_df.columns]
    )

    # Type casting for train product orders
    train_product_orders = train_product_orders.select(
        [F.col(col).cast("float").alias(col) for col in prior_product_orders.columns]
    )

    # Union of train product orders and prior product orders
    final_train_product_orders = train_product_orders.union(prior_product_orders)

    # Filter orders into training and test sets
    final_train_orders_df = orders_df.filter(F.col("eval_set") != 'test').drop('eval_set')
    test_orders_df = orders_df.filter(F.col("eval_set") == 'test').select("order_id", "user_id")

    # Feature generation
    fet_gen = FeatureGenerator(final_train_product_orders, final_train_orders_df, products_df)

    result_df = fet_gen.generate_user_related_features()
    result_prod_df = fet_gen.generate_product_related_features()
    result_user_prod_df = fet_gen.generate_user_product_related_features()
    result_time_df = fet_gen.generate_time_related_features()

    # Generate all features for the training set
    final_prior_train_set = fet_gen.generate_all_types_of_features()

    # Create test set from training data
    test_set = (
        test_orders_df.select("user_id")
            .join(final_train_orders_df, on='user_id', how='left')
            .select("user_id", "order_id")
            .join(prior_product_orders.select("order_id", "product_id"), on='order_id', how='left')
            .select("user_id", "product_id").distinct()
    )

    # Feature engineering for the test set
    featured_test_set = generate_test_set_features(result_df, result_prod_df, result_user_prod_df, result_time_df, test_set)

    # Ask user for output path
    output_path = input("Please provide the output file path (e.g., cloud storage path or local path): ")

    # Write final datasets to the output path
    final_prior_train_set.coalesce(1).write.csv(os.path.join(output_path, "final_prior_train_set.csv"))
    featured_test_set.coalesce(1).write.csv(os.path.join(output_path, "featured_test_set.csv"))

    # Save columns to text files
    with open(os.path.join(output_path, "train_set_columns.txt"), 'w') as f:
        for column in final_prior_train_set.columns:
            f.write('%s,' % column)

    with open(os.path.join(output_path, "test_set_columns.txt"), 'w') as f:
        for column in featured_test_set.columns:
            f.write('%s,' % column)


if __name__ == "__main__":
    main()
