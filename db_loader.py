import pymysql
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ["DB_PORT"])
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]


def get_connection():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    return conn


def get_all_jobs():
    """
    Lấy danh sách job + thông tin công ty (nếu có).
    """
    conn = get_connection()
    query = """
        SELECT 
            j.job_id        AS job_id,
            j.title         AS title,
            j.location      AS job_location,
            j.salary_min    AS salary_min,
            j.salary_max    AS salary_max,
            j.job_type      AS job_type,
            j.years_of_experience AS years_of_experience,
            j.education_level     AS education_level,
            j.description   AS description,
            j.benefits      AS benefits,
            j.deadline      AS deadline,
            j.posted_at     AS posted_at,
            j.is_active     AS is_active,
            j.is_approved   AS is_approved,
            j.is_deleted    AS is_deleted,
            j.is_expired    AS is_expired,
            c.company_name  AS company_name
        FROM job j
        LEFT JOIN company c ON j.company_id = c.company_id
        WHERE j.is_deleted = 0
          AND j.is_approved = 1
          AND j.is_expired = 0
    """

    # 🔹 Tự fetch bằng cursor
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    conn.close()

    # rows là list[dict] -> đưa vào DataFrame
    df = pd.DataFrame(rows)

    # ===== DEBUG =====
    print("=== DEBUG get_all_jobs() ===")
    print("Total jobs:", len(df))
    print(df[["job_id", "title", "job_location"]].head(10).to_string(index=False))

    print("\nDistinct job_location:")
    print(df["job_location"].dropna().unique())
    print("=== END DEBUG get_all_jobs() ===")

    return df



def get_summary():
    """
    Tổng quan job: tổng số job, danh sách location, job_type.
    Đảm bảo:
      - total_jobs: int
      - locations: list[str]
      - job_types: list
    """
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            # 1) Tổng job hợp lệ
            count_sql = """
                SELECT COUNT(*) AS cnt
                FROM job
                WHERE is_deleted = 0
                  AND is_approved = 1
                  AND is_expired = 0
            """
            cur.execute(count_sql)
            row = cur.fetchone()
            total_jobs = int(row["cnt"]) if row and "cnt" in row else 0

            # 2) Tất cả địa điểm (location) có job
            locations_sql = """
                SELECT DISTINCT location
                FROM job
                WHERE is_deleted = 0
                  AND is_approved = 1
                  AND is_expired = 0
                  AND location IS NOT NULL
                  AND location <> ''
                ORDER BY location
            """
            cur.execute(locations_sql)
            rows_loc = cur.fetchall()
            # rows_loc: list[dict], mỗi dict có key "location"
            locations = [r["location"] for r in rows_loc if r.get("location")]

            # 3) Các loại job_type đang dùng (0..3)
            job_type_sql = """
                SELECT DISTINCT job_type
                FROM job
                WHERE is_deleted = 0
                  AND is_approved = 1
                  AND is_expired = 0
                  AND job_type IS NOT NULL
            """
            cur.execute(job_type_sql)
            rows_type = cur.fetchall()
            job_types = [r["job_type"] for r in rows_type if r.get("job_type") is not None]

    finally:
        conn.close()

    # ===== DEBUG =====
    print("=== DEBUG get_summary() ===")
    print("total_jobs:", total_jobs, type(total_jobs))
    print("locations sample:", locations[:5], type(locations))
    print("job_types:", job_types)
    print("=== END DEBUG get_summary() ===")

    return total_jobs, locations, job_types

