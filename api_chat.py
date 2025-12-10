# chat_router.py
from fastapi import APIRouter
from pydantic import BaseModel
from chatbot import get_llm
from db_loader import get_all_jobs, get_summary
from unidecode import unidecode
import pandas as pd
import re, json

router = APIRouter()
llm = get_llm()

from vector_store import search_vector_with_filter


class ChatRequest(BaseModel):
    query: str


def preprocess(text):
    return unidecode(str(text)).lower().strip()


# ------------------- LLM EXTRACT CRITERIA ------------------- #

def extract_filter_criteria(query, llm, max_retries=3):
    """
    Rút trích tiêu chí lọc job từ câu hỏi người dùng bằng LLM.
    """
    prompt = f"""
Bạn là trợ lý tuyển dụng vui vẻ, thân thiện. Người dùng vừa hỏi:

\"{query}\"

Hãy chỉ trả về MỘT chuỗi JSON hợp lệ duy nhất, KHÔNG có giải thích gì thêm,
chứa các tiêu chí nếu có:

- location (chuỗi, ví dụ: "Hồ Chí Minh", "Hà Nội")
- job_type (chuỗi, ví dụ: "full-time", "part-time", "remote", "intern")
- position (chuỗi, ví dụ: "backend", "frontend", "tester", "data engineer")
- company (chuỗi, tên công ty, ví dụ: "FPT Software", "VNG", "Shopee")
- min_salary (số nguyên, tiền VNĐ)
- max_salary (số nguyên, tiền VNĐ)
- min_experience (số năm kinh nghiệm tối thiểu, số thực)
- max_experience (số năm kinh nghiệm tối đa, số thực)

QUAN TRỌNG:
- Nếu người dùng nói "15 triệu" hiểu là 15000000.
- Nếu người dùng nói "từ 10-15 triệu" thì:
  min_salary = 10000000, max_salary = 15000000.
- Nếu người dùng nói "1 năm kinh nghiệm", "kn 1 năm":
  min_experience = 1, max_experience = 1.
- Nếu người dùng nói "từ 1-3 năm", "1-3 năm":
  min_experience = 1, max_experience = 3.

Ví dụ trả lời:
{{
  "location": "Hà Nội",
  "position": "backend",
  "company": "FPT",
  "min_experience": 1,
  "max_experience": 1
}}

Nếu không có tiêu chí nào thì trả về {{}}

Lưu ý: CHỈ trả về JSON, không kèm thêm bất cứ văn bản nào khác.
    """

    for attempt in range(max_retries):
        response = llm.invoke(prompt)
        content = getattr(response, 'content', str(response)).strip()
        content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.MULTILINE)

        try:
            criteria = json.loads(content)
            print(f"Parsed criteria raw: {criteria}")

            # --------- CHUẨN HÓA job_type VÀ position ---------- #
            mapping = {
                "full-time", "full time",
                "part-time", "part time",
                "toàn thời gian", "toan thoi gian",
                "bán thời gian", "ban thoi gian",
                "intern", "thực tập", "thuc tap",
                "remote", "làm từ xa", "lam tu xa",
            }

            if "job_type" in criteria and criteria["job_type"]:
                jt_norm = preprocess(criteria["job_type"])
                # nếu job_type không phải các loại full-time/part-time/... → coi là position
                if jt_norm not in mapping and not jt_norm.isdigit():
                    # đẩy sang position
                    if "position" not in criteria or not criteria["position"]:
                        criteria["position"] = criteria["job_type"]
                    criteria.pop("job_type", None)

            # Chuẩn hóa min/max_experience về float nếu có
            if "min_experience" in criteria and criteria["min_experience"] is not None:
                criteria["min_experience"] = float(criteria["min_experience"])
            if "max_experience" in criteria and criteria["max_experience"] is not None:
                criteria["max_experience"] = float(criteria["max_experience"])

            print(f"Parsed criteria final: {criteria}")
            return criteria

        except json.JSONDecodeError as e:
            print(f"JSON decode error on attempt {attempt + 1}: {e}")
            print("LLM raw content:", content)
            continue

    return {}


# ------------------- LLM CLASSIFY INTENT ------------------- #

def analyze_query_with_llm(query):
    """
    Phân loại loại câu hỏi:
    - summary: hỏi tổng số job, thống kê
    - search: tìm job theo lương / địa điểm / loại việc / kinh nghiệm
    - advice: nhờ tư vấn định hướng, gợi ý ngành / vị trí
    """
    prompt = f"""
Bạn là một AI thông minh, thân thiện, hiểu tiếng Việt.

Người dùng hỏi:
"{query}"

Hãy phân loại câu hỏi này vào đúng nhóm sau (chỉ trả về đúng 1 từ):
- summary → nếu hỏi tổng số công việc, số lượng job, thống kê
- search  → nếu tìm kiếm việc làm theo tiêu chí (lương, địa điểm, kinh nghiệm, loại job,…)
- advice  → nếu người dùng nhờ tư vấn định hướng nghề nghiệp, gợi ý vị trí phù hợp

Trả về đúng 1 từ: summary, search hoặc advice.
"""
    response = llm.invoke(prompt)
    action = response.content.strip().lower()
    print(f"[analyze_query_with_llm] Query: {query} => Action: {action}")
    return action


# ------------------- NORMALIZE LOCATION & EXPERIENCE ------------------- #

def normalize_location(text: str) -> str:
    """
    Chuẩn hóa location để so sánh:
    - Bỏ dấu, lower
    - Bỏ 'tp.', 'thanh pho', 'thành phố'
    - Map các alias như: hanoi, hn -> ha noi
    """
    if text is None:
        return ""

    p = preprocess(text)
    # bỏ tiền tố thành phố
    for prefix in ["thanh pho ", "thành phố ", "tp. ", "tp ", "tp"]:
        if p.startswith(prefix):
            p = p[len(prefix):]

    # bỏ chữ ', viet nam' nếu có
    p = p.replace(", viet nam", "").replace(", vietnam", "")

    # alias phổ biến
    aliases = {
        "ha noi": "ha noi",
        "hanoi": "ha noi",
        "hn": "ha noi",
        "ha noi,": "ha noi",

        "ho chi minh": "ho chi minh",
        "hcm": "ho chi minh",
        "tphcm": "ho chi minh",
        "tp ho chi minh": "ho chi minh",
        "sai gon": "ho chi minh",
    }

    # nếu trùng key thì trả key chuẩn
    if p in aliases:
        return aliases[p]

    # nếu chuỗi dài kiểu "quan cau giay, ha noi" thì giữ nguyên nhưng đã normalize
    return p


def parse_years_of_exp(value):
    """
    Chuyển '1', '2', '2-3' -> (min_years, max_years)
    """
    if pd.isna(value):
        return (None, None)

    s = str(value).strip()
    if not s:
        return (None, None)

    if "-" in s:
        p1, p2 = s.split("-", 1)
        try:
            lo = float(p1.strip())
        except ValueError:
            lo = None
        try:
            hi = float(p2.strip())
        except ValueError:
            hi = None
    else:
        try:
            lo = hi = float(s)
        except ValueError:
            lo = hi = None

    return (lo, hi)


# ------------------- MAIN FILTER FUNCTION ------------------- #

def filter_jobs(df, criteria):
    filtered = df.copy()

    # ===== LOCATION =====
    if "location" in criteria and criteria["location"]:
        raw_loc = criteria["location"]
        loc_norm = normalize_location(raw_loc)
        print(">>> Filter by location (raw):", raw_loc, "| (norm):", loc_norm)

        def match_location(x):
            if x is None:
                return False
            xp_norm = normalize_location(x)
            # match nếu giống nhau hoặc chứa nhau
            return (
                loc_norm == xp_norm
                or loc_norm in xp_norm
                or xp_norm in loc_norm
            )

        filtered = filtered[filtered["job_location"].apply(match_location)]
        print(">>> After location filter:", len(filtered))

        # Fallback: nếu vẫn 0 thì thử tìm theo text trong title/description
        if len(filtered) == 0:
            q = preprocess(raw_loc)
            print(">>> Fallback search in title/description with:", q)

            def match_text(row):
                title = preprocess(row.get("title", ""))
                desc = preprocess(row.get("description", ""))
                loc = preprocess(row.get("location", ""))
                return q in title or q in desc or q in loc

            filtered = df[df.apply(match_text, axis=1)]
            print(">>> After fallback text search:", len(filtered))

    # ===== COMPANY =====
    if "company" in criteria and criteria["company"]:
        company_raw = criteria["company"]
        company_kw = preprocess(company_raw)
        print(">>> Filter by company (raw):", company_raw, "| norm:", company_kw)

        if "company_name" in filtered.columns:
            def match_company(x):
                if not x:
                    return False
                name_norm = preprocess(x)
                # match nếu chứa nhau hai chiều (FPT vs fpt software)
                return company_kw in name_norm or name_norm in company_kw

            before = len(filtered)
            filtered = filtered[filtered["company_name"].apply(match_company)]
            print(f">>> Filter by company => {before} -> {len(filtered)}")
        else:
            print(">>> WARNING: 'company_name' column not found in df, skip company filter")

    # ===== JOB_TYPE + POSITION / ROLE KEYWORD =====
    role_kw = None  # keyword nghề để search title/description (backend, frontend,...)

    # 2.1. Nếu có position => dùng làm role_kw
    if "position" in criteria and criteria["position"]:
        role_kw = preprocess(criteria["position"])

    # 2.2. Xử lý job_type (full-time, part-time, intern, remote hoặc số)
    if "job_type" in criteria and criteria["job_type"]:
        jt_raw = criteria["job_type"]
        jt = preprocess(jt_raw)

        mapping = {
            "full-time": 0,
            "full time": 0,
            "toan thoi gian": 0,
            "toàn thời gian": 0,
            "part-time": 1,
            "part time": 1,
            "ban thoi gian": 1,
            "bán thời gian": 1,
            "intern": 2,
            "thuc tap": 2,
            "thực tập": 2,
            "remote": 3,
            "lam tu xa": 3,
            "làm từ xa": 3,
        }

        if jt in mapping:
            # lọc theo cột job_type (0–3)
            before = len(filtered)
            filtered = filtered[filtered["job_type"] == mapping[jt]]
            print(f">>> Filter by job_type '{jt}' => {before} -> {len(filtered)}")
        elif jt.isdigit():
            before = len(filtered)
            filtered = filtered[filtered["job_type"] == int(jt)]
            print(f">>> Filter by job_type digit '{jt}' => {before} -> {len(filtered)}")
        # còn trường hợp jt là "backend" thì ở bước extract mình đã chuyển nó sang position rồi

    # 2.3. Nếu có role_kw (backend/frontend/...) thì lọc theo title/description
    if role_kw:
        def match_role(row):
            title = preprocess(row.get("title", "")) or ""
            desc = preprocess(row.get("description", "")) or ""
            return role_kw in title or role_kw in desc

        before = len(filtered)
        filtered = filtered[filtered.apply(match_role, axis=1)]
        print(f">>> Filter by role keyword '{role_kw}' => {before} -> {len(filtered)}")

    # ===== SALARY =====
    if "min_salary" in criteria and criteria["min_salary"] is not None:
        before = len(filtered)
        filtered = filtered[filtered["salary_min"] >= criteria["min_salary"]]
        print(f">>> Filter by min_salary {criteria['min_salary']} => {before} -> {len(filtered)}")

    if "max_salary" in criteria and criteria["max_salary"] is not None:
        before = len(filtered)
        filtered = filtered[filtered["salary_max"] <= criteria["max_salary"]]
        print(f">>> Filter by max_salary {criteria['max_salary']} => {before} -> {len(filtered)}")

    # ===== EXPERIENCE =====
    has_min_exp = "min_experience" in criteria and criteria["min_experience"] is not None
    has_max_exp = "max_experience" in criteria and criteria["max_experience"] is not None

    if has_min_exp or has_max_exp:
        min_exp = float(criteria.get("min_experience")) if has_min_exp else None
        max_exp = float(criteria.get("max_experience")) if has_max_exp else None

        def match_experience(row):
            val = row.get("years_of_experience")   # cột trong DB: '1', '2', '2-3'
            lo, hi = parse_years_of_exp(val)

            if lo is None and hi is None:
                return False

            if has_min_exp and hi is not None and hi < min_exp:
                return False
            if has_max_exp and lo is not None and lo > max_exp:
                return False

            return True

        before = len(filtered)
        filtered = filtered[filtered.apply(match_experience, axis=1)]
        print(f">>> Filter by experience ({min_exp} - {max_exp}) => {before} -> {len(filtered)}")

    print(">>> Filtered jobs count (final):", len(filtered))
    return filtered


def generate_career_advice(user_input: str, df_jobs: pd.DataFrame, llm):
    """
    Sinh câu trả lời tư vấn hướng nghiệp (advice) bằng LLM.
    Có thể dùng danh sách job hiện có làm dữ liệu tham khảo.
    """
    # Lấy 10–15 job title làm context gợi ý
    try:
        unique_titles = df_jobs["title"].dropna().unique().tolist()
        sample_titles = unique_titles[:15]
        titles_text = "\n".join(f"- {t}" for t in sample_titles)
    except Exception:
        titles_text = ""

    prompt = f"""
Bạn là chuyên gia hướng nghiệp thân thiện, nói chuyện như một anh/chị mentor đi làm lâu năm.

Người dùng vừa chia sẻ:
\"\"\"{user_input}\"\"\"


Dưới đây là một số vị trí đang phổ biến trên thị trường (chỉ để bạn tham khảo, đừng liệt kê lại y chang):
{titles_text}

YÊU CẦU CÁCH TRẢ LỜI:
- Trả lời BẰNG TIẾNG VIỆT.
- Giọng văn gần gũi, khích lệ, nhưng thực tế.
- Giải thích giúp người dùng:
  1) Nên cân nhắc theo hướng nào (2–3 lựa chọn chính, ví dụ: frontend, backend, tester, data, BA,…),
  2) Mỗi hướng phù hợp với kiểu người / điểm mạnh nào,
  3) Gợi ý lộ trình học ngắn gọn (kỹ năng, công nghệ nên học),
  4) Một vài lời khuyên thực tế (làm project nhỏ, thực tập, CV, phỏng vấn).

KHÔNG được trả lời kiểu chung chung 1–2 câu. Hãy trả lời chi tiết, có cấu trúc, nhưng vẫn dễ đọc.
    """

    res = llm.invoke(prompt)
    content = getattr(res, "content", str(res))
    return content.strip()



# ------------------- MAIN ROUTE ------------------- #

@router.post("/chat")
def chat(req: ChatRequest):
    user_input = req.query
    action = analyze_query_with_llm(user_input)
    print(f"Action: {action}, Query: {user_input}")

    df_jobs = get_all_jobs()
    total_jobs, locations, job_types = get_summary()

    # --- SUMMARY ---
    if action == "summary":
        criteria = extract_filter_criteria(user_input, llm)

        # Nếu có tiêu chí lương / location / job_type / company thì thống kê theo filter
        if any(k in criteria for k in ["min_salary", "max_salary", "location", "job_type", "company"]):
            filtered = filter_jobs(df_jobs, criteria)
            count = len(filtered)

            if count == 0:
                reply = "Hiện chưa có công việc nào phù hợp với tiêu chí bạn đưa ra."
            else:
                preview = filtered.head(5)
                job_list = "\n".join([
                    f"- {row['title']} tại {row.get('company_name') or 'Công ty'} "
                    f"({row['job_location']}), Lương: {int(row['salary_min']):,} - {int(row['salary_max']):,} VNĐ"
                    for _, row in preview.iterrows()
                ])
                reply = (
                    f"Có tổng cộng **{count}** việc làm phù hợp với tiêu chí của bạn.\n"
                    f"Một vài gợi ý:\n{job_list}"
                )
        else:
            reply = (
                f"Hiện hệ thống đang có **{total_jobs}** tin tuyển dụng đang hoạt động.\n"
                f"Một số khu vực có nhiều việc làm: {', '.join(locations[:5])}."
            )

    # --- SEARCH ---
    elif action == "search":
        criteria = extract_filter_criteria(user_input, llm)
        print(f"Criteria extracted: {criteria}")

        filtered_jobs = filter_jobs(df_jobs, criteria)
        count = len(filtered_jobs)

        if count == 0:
            # fallback sang vector search
            try:
                pinecone_filter = {}
                if "location" in criteria and criteria["location"]:
                    pinecone_filter["location"] = {"$eq": criteria["location"]}
                if "company" in criteria and criteria["company"]:
                    pinecone_filter["company_name"] = {"$eq": criteria["company"]}

                vector_answer = search_vector_with_filter(
                    query=user_input,
                    llm=llm,
                    filter_dict=pinecone_filter or None,
                    top_k=5
                )
                reply = (
                    "Dữ liệu filter theo cấu trúc chưa tìm thấy job phù hợp.\n"
                    "Tuy nhiên mình có vài gợi ý thông minh dựa trên nội dung JD:\n"
                    f"{vector_answer}"
                )
            except Exception as e:
                print("Vector search error:", e)
                suggestions = ", ".join(locations[:5])
                reply = (
                    "Mình chưa tìm thấy công việc nào phù hợp với yêu cầu của bạn.\n"
                    f"Bạn có thể nói rõ hơn về vị trí + địa điểm + mức lương mong muốn.\n"
                    f"Một số khu vực đang có nhiều job: {suggestions}."
                )
        else:
            preview = filtered_jobs.head(5)
            job_list = "\n".join([
                f"- {row['title']} tại {row.get('company_name') or 'Công ty'} "
                f"({row['job_location']}), Lương: {int(row['salary_min']):,} - {int(row['salary_max']):,} VNĐ"
                for _, row in preview.iterrows()
            ])
            reply = (
                f"Mình tìm được **{count}** việc làm phù hợp với yêu cầu của bạn.\n"
                f"Bạn tham khảo trước vài job sau nhé:\n{job_list}\n"
                "Nếu cần, bạn có thể nói rõ hơn (vị trí, kinh nghiệm, mức lương mong muốn) để mình lọc kỹ hơn."
            )

    # --- ADVICE ---
    elif action == "advice":
        print(">>> Going to advice mode (no job filtering)")
        reply = generate_career_advice(user_input, df_jobs, llm)

    else:
        reply = (
            "Mình chưa hiểu câu hỏi lắm, bạn có thể nói rõ hơn không?\n"
            "Ví dụ: \"Mình muốn tìm việc IT ở TP.HCM, lương từ 15 tới 20 triệu\"."
        )

    return {"answer": reply}
