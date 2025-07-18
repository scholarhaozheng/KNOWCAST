import mysql.connector
import datetime

combined_table = False
date_filter_need = False
date_filter_and_combine = True

host = ''
user = ''
password = ''
database = ''
mother_table =''

filter_dates = ['']

conn = mysql.connector.connect(
    host='',
    user='',
    password='',
    database='',
)

print(conn)
print(type(conn))

cursor = conn.cursor()

cursor.execute(f"SELECT * FROM {mother_table} LIMIT 10")
data = cursor.fetchall()


try:
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    if conn.is_connected():
        print("")

    cursor = conn.cursor()

    cursor.execute(f"DESCRIBE {mother_table}")
    table_structure = cursor.fetchall()
    print("：")
    for col in table_structure:
        print(col)

    cursor.execute(f"SELECT DISTINCT data_date FROM {mother_table} LIMIT 10")
    dates = cursor.fetchall()
    print("\n：", dates)

    if combined_table:
        combined_table_name = ""

        insert_combined_query = f"""
            INSERT INTO {combined_table_name}
            SELECT * FROM 
            UNION ALL
            SELECT * FROM 
            """
        cursor.execute(insert_combined_query)
        conn.commit()
        print(f" '{combined_table_name}' ")

        cursor.execute(f"DROP TABLE IF EXISTS {combined_table_name}")
        cursor.execute(f"CREATE TABLE {combined_table_name} LIKE ")

    if date_filter_need:
        for dt in filter_dates:
            new_table_name = f"{mother_table}_filtered_{dt.replace('-', '_')}"
            print(f"\n: {dt}, : {new_table_name}")

            cursor.execute(f"DROP TABLE IF EXISTS {new_table_name}")
            cursor.execute(f"CREATE TABLE {new_table_name} LIKE {mother_table}")

            insert_query = f"""
            INSERT INTO {new_table_name}
            SELECT * FROM {mother_table}
            WHERE data_date = '{dt}'
            """
            cursor.execute(insert_query)
            conn.commit()

            print(f" '{new_table_name}' {dt} 。")

    if date_filter_and_combine:
        def extract_filtered_data(cursor, conn, source_table_name, new_table_name, start_date, end_date):
            print(f"\n {source_table_name}  {start_date}  {end_date}{new_table_name}")

            drop_table_query = f"DROP TABLE IF EXISTS {new_table_name}"
            cursor.execute(drop_table_query)

            create_table_query = f"CREATE TABLE {new_table_name} LIKE {source_table_name}"
            cursor.execute(create_table_query)

            insert_filtered_query = f"""
                INSERT INTO {new_table_name}
                SELECT * FROM {source_table_name}
                WHERE data_date BETWEEN '{start_date}' AND '{end_date}'
            """
            cursor.execute(insert_filtered_query)

            conn.commit()


        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="",
            new_table_name="",
            start_date="",
            end_date=""
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="",
            new_table_name="",
            start_date="",
            end_date=""
        )


except mysql.connector.Error as err:
    print(":", err)

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("\n。")
