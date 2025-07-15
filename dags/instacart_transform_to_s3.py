import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import boto3

# S3 configuration
S3_BUCKET = 'instacart_basket_recommandation'
S3_OUTPUT_PREFIX = 'output/'
LOCAL_OUTPUT_DIR = 'output/'

# Input file paths (relative to project root)
ORDERS_FILE = 'data/orders.csv'
PRIOR_PRODUCT_ORDERS_FILE = 'data/order_products__prior.csv'
PRODUCTS_FILE = 'data/products.csv'
TRAIN_PRODUCT_ORDERS_FILE = 'data/order_products__train.csv'

# Function to upload all files in output/ to S3
def upload_output_to_s3(**kwargs):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(LOCAL_OUTPUT_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = os.path.join(S3_OUTPUT_PREFIX, file)
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'instacart_transform_to_s3',
    default_args=default_args,
    description='Run Instacart transformation and upload to S3',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Task 1: Run the transformation script
run_transform = BashOperator(
    task_id='run_transformation',
    bash_command=(
        'mkdir -p output && '
        'ORDERS_FILE_PATH={{params.orders}} '
        'PRIOR_PRODUCT_ORDERS_FILE_PATH={{params.prior}} '
        'PRODUCTS_FILE_PATH={{params.products}} '
        'TRAIN_PRODUCT_ORDERS_FILE_PATH={{params.train}} '
        'python3 src/final_dataset_generator.py <<EOF\noutput\nEOF'
    ),
    params={
        'orders': ORDERS_FILE,
        'prior': PRIOR_PRODUCT_ORDERS_FILE,
        'products': PRODUCTS_FILE,
        'train': TRAIN_PRODUCT_ORDERS_FILE,
    },
    dag=dag,
)

# Task 2: Upload output files to S3
upload_to_s3 = PythonOperator(
    task_id='upload_output_to_s3',
    python_callable=upload_output_to_s3,
    dag=dag,
)

run_transform >> upload_to_s3 