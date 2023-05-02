import pandas as pd
import numpy as np
import os
import sys
import sqlite3
from datetime import datetime

def get_olist_data_as_df(db_file, from_purchase_date=None, to_purchase_date=None, conn=None):

    local_conn = False
    if conn is None:
        conn = sqlite3.connect(db_file)
        local_conn = True

    if from_purchase_date is None:
        from_purchase_date = "2001-01-01 00:00:00"

    if to_purchase_date is None:
        to_purchase_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sql = """DROP TABLE IF EXISTS tt"""
    conn.execute(sql)
    sql = """CREATE TABLE tt AS 
        SELECT 
            c.customer_unique_id AS customer_unique_id, 
            o.order_id AS order_id, 
            SUM(oi.price) AS purchase_amount, 
            SUM(oi.freight_value) AS freight_value,
            SUM(p.product_photos_qty) AS photos_quantity,
            GROUP_CONCAT(tcnt.product_category_name_english||':'||oi.price) AS categories,
            GROUP_CONCAT(op.payment_type||':'||op.payment_value) AS payments,
            COUNT(*) AS nb_items, 
            o.order_purchase_timestamp AS purchase_timestamp,
            (JULIANDAY(o.order_estimated_delivery_date) - JULIANDAY(o.order_purchase_timestamp)) AS expected_delivery_delay,
            (JULIANDAY(o.order_delivered_customer_date) - JULIANDAY(o.order_purchase_timestamp)) AS effective_delivery_delay
        FROM customers c 
            LEFT JOIN orders o ON c.customer_id = o.customer_id 
            LEFT JOIN order_items oi ON oi.order_id = o.order_id
            LEFT JOIN products p ON p.product_id = oi.product_id 
            LEFT JOIN t_category_name_translation tcnt ON tcnt.product_category_name = p.product_category_name 
            LEFT JOIN order_payments op ON op.order_id=o.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable') AND o.order_purchase_timestamp<=? AND o.order_purchase_timestamp>=?
        GROUP BY customer_unique_id, o.order_id"""
    conn.execute(sql, (to_purchase_date, from_purchase_date,))

    conn.execute("CREATE INDEX tt_customer_unique_id_IDX ON tt(customer_unique_id);")
    conn.execute("CREATE INDEX tt_order_id_IDX ON tt(order_id);")

    sql = """
    WITH customers_geo (customer_unique_id, latitude, longitude, state) AS (
        SELECT c2.customer_unique_id, g.latitude  , g.longitude, g.state  FROM customers c2, geoloc g WHERE c2.customer_state = g.state AND instr(c2.customer_zip_code_prefix,g.zip_code_prefix) = 1   
    ),
    all_customers (
        customer_unique_id, last_order_id, last_amount, last_freight_value, last_photos_quantity, last_categories, last_payments, last_nb_items, last_timestamp, 
        last_expected_delivery_delay, last_effective_delivery_delay, 
        last_review_score, all_purchases_amount, average_amount, nb_all_purchases, all_purchases_review_score, all_purchases_categories, all_purchases_payments, 
        all_purchases_timestamps, customer_state, customer_latitude, customer_longitude,
        all_expected_delivery_delay, all_effective_delivery_delay, all_freight_value, all_photos_quantity) AS 
    (
        SELECT tt.*, 
        (
            SELECT AVG(or2.review_score) FROM order_reviews or2 WHERE or2.order_id = tt.order_id
        ) AS review_score,
        (
            SELECT sum(oi2.price)
            FROM customers c2 
                LEFT JOIN orders o2 ON c2.customer_id = o2.customer_id 
                LEFT JOIN order_items oi2 ON o2.order_id = oi2.order_id  
            WHERE c2.customer_unique_id = tt.customer_unique_id
        ) AS all_purchases_amount,
        (
            SELECT AVG(purchase_amount) FROM tt tt2 WHERE tt2.customer_unique_id=tt.customer_unique_id
        ) AS average_amount,
        (
            SELECT count(*) FROM tt tt2 WHERE tt2.customer_unique_id=tt.customer_unique_id
        ) AS nb_purchases,
        (
            SELECT AVG(or3.review_score) 
            FROM order_reviews or3, orders o3, customers c3 
            WHERE o3.order_id = or3.order_id AND c3.customer_id = o3.customer_id AND c3.customer_unique_id  = tt.customer_unique_id
        ) AS all_purchases_review_score,
        (
            SELECT GROUP_CONCAT(tcnt2.product_category_name_english||':'||oi3.price) AS categories
            FROM t_category_name_translation tcnt2, products p2 , order_items oi3 , orders o4 , customers c4  
            WHERE tcnt2.product_category_name=p2.product_category_name AND p2.product_id=oi3.product_id AND oi3.order_id =o4.order_id AND o4.customer_id = c4.customer_id AND c4.customer_unique_id=tt.customer_unique_id
        ) AS all_purchases_categories,
        (
            SELECT GROUP_CONCAT(op2.payment_type||':'||op2.payment_value) AS payments
            FROM order_payments op2 join orders o5, customers c5
            WHERE op2.order_id =o5.order_id AND o5.customer_id = c5.customer_id AND c5.customer_unique_id=tt.customer_unique_id
        ) AS all_purchases_payments,
        (
            SELECT GROUP_CONCAT(tt3.purchase_timestamp) AS purchases_timestamps
            FROM tt tt3
            WHERE tt3.customer_unique_id=tt.customer_unique_id
        ) AS all_purchases_timestamps,
        (
            SELECT state FROM customers_geo WHERE customer_unique_id = tt.customer_unique_id
        ) AS state, 
        (
            SELECT latitude FROM customers_geo WHERE customer_unique_id = tt.customer_unique_id
        ) AS latitude, 
        (
            SELECT longitude FROM customers_geo WHERE customer_unique_id = tt.customer_unique_id
        ) AS longitude,
        (
            SELECT AVG(expected_delivery_delay) FROM tt tt2 WHERE tt2.customer_unique_id=tt.customer_unique_id
        ) AS all_expected_delivery_delay,
        (
            SELECT AVG(effective_delivery_delay) FROM tt tt2 WHERE tt2.customer_unique_id=tt.customer_unique_id
        ) AS all_effective_delivery_delay,
        (
            SELECT AVG(freight_value) FROM tt tt2 WHERE tt2.customer_unique_id=tt.customer_unique_id
        ) AS all_freight_value,
        (
            SELECT AVG(photos_quantity) FROM tt tt2 WHERE tt2.customer_unique_id=tt.customer_unique_id
        ) AS all_photos_quantity
        from tt 
        order by customer_unique_id , purchase_timestamp desc
    ) 
    SELECT customer_unique_id,last_order_id,customer_state,customer_latitude,customer_longitude,nb_all_purchases,average_amount,
        last_timestamp,last_nb_items,last_categories,last_amount, last_expected_delivery_delay, last_effective_delivery_delay, last_freight_value, last_photos_quantity, 
        last_payments,last_review_score,all_purchases_timestamps,all_purchases_categories,
        all_purchases_amount,all_purchases_payments, all_purchases_review_score, all_expected_delivery_delay, all_effective_delivery_delay , all_freight_value, all_photos_quantity
    FROM all_customers 
    WHERE last_order_id IN (
        SELECT MAX(last_order_id) from all_customers group by customer_unique_id 
    )
    """

    df = pd.read_sql_query(sql, conn)

    if local_conn:
        conn.close()

    return df



def get_olist_dataframe(db_file=None, from_purchase_date=None, to_purchase_date=None, dir_path="assets/olist"):
    
    working_dir = os.path.dirname(db_file)
    if not os.path.exists(db_file):
        open(db_file, "a").close()
        conn = sqlite3.connect(db_file)
        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)) and path.lower().endswith(
                ".csv"
            ):
                df = pd.read_csv(os.path.join(dir_path, path))
                df.drop_duplicates(inplace=True)
                table_name = path[0:-4]
                table_name = table_name[6:]
                indexes = []
                for c in df.columns:
                    if c.endswith("_id"):
                        indexes.append(c)
                if table_name.endswith("_dataset"):
                    table_name = table_name[0:-8]
                print(f"{table_name} indexes={indexes}")
                df.to_sql(
                    table_name, conn, if_exists="replace", index=False, chunksize=200
                )
                for ind in indexes:
                    conn.execute(
                        f"""CREATE INDEX {table_name}_{ind}_IDX ON {table_name}({ind});"""
                    )

        conn.execute(
            "CREATE INDEX customers_customer_zip_code_prefix_IDX ON customers(customer_zip_code_prefix);"
        )
        conn.execute(
            "CREATE INDEX customers_customer_state_IDX ON customers(customer_state);"
        )
        conn.execute(
            "CREATE INDEX geolocation_geolocation_zip_code_prefix_IDX ON geolocation(geolocation_zip_code_prefix);"
        )
        conn.execute(
            "CREATE INDEX orders_order_purchase_timestamp_IDX ON orders(order_purchase_timestamp);"
        )
        conn.execute(
            """CREATE TABLE geoloc 
            AS SELECT g.geolocation_zip_code_prefix AS zip_code_prefix, g.geolocation_state AS state, AVG(g.geolocation_lat) AS latitude, AVG(g.geolocation_lng) AS longitude
            FROM geolocation g GROUP BY geolocation_zip_code_prefix , geolocation_state  """
        )
        conn.execute("CREATE INDEX geoloc_state_IDX ON geoloc(state);")
        conn.execute(
            "CREATE UNIQUE INDEX geoloc_zip_code_prefix_IDX ON geoloc (zip_code_prefix,state);"
        )
        conn.close()
        
    if from_purchase_date is None and to_purchase_date is None:
        if not os.path.exists(f"{working_dir}/olist.parquet"):
            df = get_olist_data_as_df(db_file=db_file)
            df.to_parquet(f"{working_dir}/olist.parquet")
        else:
            df = pd.read_parquet(f"{working_dir}/olist.parquet")
    else:
        df = get_olist_data_as_df(db_file=db_file, from_purchase_date=from_purchase_date, to_purchase_date=to_purchase_date)

    return df