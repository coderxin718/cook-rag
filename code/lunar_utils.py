"""
农历与节气工具模块
纯 Python 实现，零外部依赖。用于「每日推荐」功能的日期上下文展示。

功能:
  - 公历 → 农历日期转换（2020–2035 年）
  - 24 节气查询（近似日期 ±1 天精度）
  - 干支纪年（天干地支）
  - 格式化日期上下文字符串
"""

from datetime import date, timedelta
from typing import Optional

# ============================================================================
# 农历数据 (2020–2035)
# ============================================================================
# 每条记录: (公历年份, 正月初一公历月, 正月初一公历日, 闰月(0=无), 月份天数编码)
# 月份天数编码: 低12位依次表示第1–12个月, 1=大月(30天), 0=小月(29天)
#               若闰月非0, 则 bit12 表示闰月天数 (1=大, 0=小)
# 数据参考香港天文台农历年历
_LUNAR_DATA = [
    # (year, ny_month, ny_day, leap, encoding)
    (2020,  1, 25,  4, 0x054BD8),   # 庚子年 闰四月小
    (2021,  2, 12,  0, 0x0494E0),   # 辛丑年
    (2022,  2,  1,  0, 0x0A9550),   # 壬寅年
    (2023,  1, 22,  2, 0x0554D5),   # 癸卯年 闰二月小
    (2024,  2, 10,  0, 0x0D2A60),   # 甲辰年
    (2025,  1, 29,  6, 0x0D9524),   # 乙巳年 闰六月小
    (2026,  2, 17,  0, 0x0D52A0),   # 丙午年
    (2027,  2,  6,  0, 0x0AA550),   # 丁未年
    (2028,  1, 26,  5, 0x056D52),   # 戊申年 闰五月小
    (2029,  2, 13,  0, 0x0AAAE0),   # 己酉年
    (2030,  2,  3,  0, 0x0A5520),   # 庚戌年
    (2031,  1, 23,  3, 0x04B558),   # 辛亥年 闰三月小
    (2032,  2, 11,  0, 0x0494E0),   # 壬子年
    (2033,  1, 31,  7, 0x0EA524),   # 癸丑年 闰七月小
    (2034,  2, 19,  0, 0x0D2AA0),   # 甲寅年
    (2035,  2,  8,  0, 0x0A9500),   # 乙卯年
]

# 天干地支
_TIANGAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
_DIZHI   = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
_SHENGXIAO = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]

# 农历月份名
_LUNAR_MONTHS = ["", "正月", "二月", "三月", "四月", "五月", "六月",
                       "七月", "八月", "九月", "十月", "冬月", "腊月"]
_LUNAR_DAYS = ["", "初一", "初二", "初三", "初四", "初五", "初六", "初七", "初八", "初九", "初十",
                     "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
                     "廿一", "廿二", "廿三", "廿四", "廿五", "廿六", "廿七", "廿八", "廿九", "三十"]


def _build_lunar_days(year_idx: int) -> list[int]:
    """构建指定农历年的每月天数列表（含闰月）"""
    _, _, _, leap, encoding = _LUNAR_DATA[year_idx]
    months = []
    for i in range(12):
        months.append(30 if (encoding >> i) & 1 else 29)
    if leap:
        leap_days = 30 if (encoding >> 12) & 1 else 29
        months.insert(leap, leap_days)  # 闰月插在该月之后
    return months


def _greg_to_lunar(d: date) -> Optional[dict]:
    """公历日期转农历日期"""
    for idx, (year, ny_m, ny_d, *_rest) in enumerate(_LUNAR_DATA):
        ny_date = date(year, ny_m, ny_d)
        # 找到下一个农历新年的日期
        if idx + 1 < len(_LUNAR_DATA):
            next_ny = date(_LUNAR_DATA[idx + 1][0],
                           _LUNAR_DATA[idx + 1][1],
                           _LUNAR_DATA[idx + 1][2])
        else:
            # 超出数据范围，假设下一年在类似日期
            next_ny = date(year + 1, ny_m, ny_d)

        if not (ny_date <= d < next_ny):
            continue

        # d 落在该农历年内
        month_days = _build_lunar_days(idx)
        offset = (d - ny_date).days

        for mi, md in enumerate(month_days):
            if offset < md:
                # 判断是否为闰月
                _, _, _, leap, _enc = _LUNAR_DATA[idx]
                is_leap = leap != 0 and mi == leap
                # 闰月插在原月之后，所以实际月号是 mi（若 mi <= leap）或 mi（若 mi > leap）
                # 构建月索引映射：month_days 中，若 mi < leap，月号=mi+1; 若 mi==leap，月号=leap(闰); 若 mi>leap，月号=mi
                if leap and mi == leap:
                    lunar_month = leap
                    is_leap = True
                elif leap and mi > leap:
                    lunar_month = mi
                    is_leap = False
                else:
                    lunar_month = mi + 1
                    is_leap = False

                lunar_day = offset + 1
                return {
                    "lunar_year": year,
                    "lunar_month": lunar_month,
                    "lunar_day": lunar_day,
                    "is_leap": is_leap,
                    "month_name": ("闰" if is_leap else "") + _LUNAR_MONTHS[lunar_month],
                    "day_name": _LUNAR_DAYS[lunar_day],
                }
            offset -= md

    return None  # 超出数据范围


def get_tiangan_dizhi(year: int) -> str:
    """获取干支纪年（如 丙午）"""
    base = 1984  # 甲子年
    offset = (year - base) % 60
    return _TIANGAN[offset % 10] + _DIZHI[offset % 12]


def get_shengxiao(year: int) -> str:
    """获取生肖"""
    return _SHENGXIAO[(year - 4) % 12]


# ============================================================================
# 24节气 (近似日期, 2000–2100 精度 ±1 天)
# ============================================================================
_SOLAR_TERMS = [
    ("小寒", 1, 5),   ("大寒", 1, 20),  ("立春", 2, 4),   ("雨水", 2, 19),
    ("惊蛰", 3, 6),   ("春分", 3, 21),  ("清明", 4, 5),   ("谷雨", 4, 20),
    ("立夏", 5, 5),   ("小满", 5, 21),  ("芒种", 6, 6),   ("夏至", 6, 21),
    ("小暑", 7, 7),   ("大暑", 7, 23),  ("立秋", 8, 7),   ("处暑", 8, 23),
    ("白露", 9, 8),   ("秋分", 9, 23),  ("寒露", 10, 8),  ("霜降", 10, 23),
    ("立冬", 11, 7),  ("小雪", 11, 22), ("大雪", 12, 7),  ("冬至", 12, 22),
]


def _term_date(term: tuple, year: int) -> date:
    """获取某年某节气的公历日期"""
    _name, month, day = term
    return date(year, month, day)


def get_solar_term_context(d: date) -> str:
    """
    获取日期的节气上下文描述。

    Returns:
        如 "夏至后第4天"、"今日夏至"、"小暑前第12天"
    """
    year = d.year

    # 构建该年所有节气日期（含前后年份的冬至/小寒以处理年初年末）
    terms_of_year = []
    for term in _SOLAR_TERMS:
        terms_of_year.append((term[0], _term_date(term, year)))

    # 按日期排序
    terms_of_year.sort(key=lambda x: x[1])

    # 找到最近的已过节气
    prev_term = None
    next_term = None
    for name, td in terms_of_year:
        if td <= d:
            prev_term = (name, td)
        if td >= d and next_term is None:
            next_term = (name, td)

    # 处理跨年情况：如果日期在年初（如1月1日）且没有 prev_term
    if prev_term is None:
        # 找去年最后一个节气（冬至）
        prev_term = ("冬至", _term_date(
            next(t for t in _SOLAR_TERMS if t[0] == "冬至"), year - 1
        ))

    if prev_term and prev_term[1] == d:
        return f"今日{prev_term[0]}"

    if prev_term:
        diff = (d - prev_term[1]).days
        if diff <= 15:
            return f"{prev_term[0]}后第{diff}天"

        # 如果离下一个更近，用"xxx前"
        if next_term:
            diff_next = (next_term[1] - d).days
            if diff_next < diff and diff_next <= 15:
                return f"{next_term[0]}前第{diff_next}天"

        return f"{prev_term[0]}后第{diff}天"

    return ""


def get_nearest_solar_term(d: date) -> str:
    """获取日期最近的节气名称"""
    year = d.year
    best_term = None
    best_diff = 999

    for term in _SOLAR_TERMS:
        td = _term_date(term, year)
        diff = abs((d - td).days)
        if diff < best_diff:
            best_diff = diff
            best_term = term[0]

    return best_term or ""


# ============================================================================
# 季节判断
# ============================================================================
def get_season(d: date) -> str:
    """根据日期判断季节"""
    m, day = d.month, d.day
    if (m == 3 and day >= 21) or m in (4, 5) or (m == 6 and day < 21):
        return "春季"
    elif (m == 6 and day >= 21) or m in (7, 8) or (m == 9 and day < 23):
        return "夏季"
    elif (m == 9 and day >= 23) or m in (10, 11) or (m == 12 and day < 22):
        return "秋季"
    else:
        return "冬季"


# ============================================================================
# 格式化输出
# ============================================================================
# 星期映射
_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def format_date_context(d: date = None) -> dict:
    """
    生成日期上下文字典，供每日推荐使用。

    Returns:
        {
            "date": "2026-06-25",
            "weekday": "星期四",
            "lunar_full": "农历五月初十",
            "lunar_year_name": "丙午年",
            "solar_term_context": "夏至后第4天",
            "season": "夏季",
            "display_date": "2026年6月25日 星期四",
            "display_lunar": "丙午年 五月初十",
        }
    """
    if d is None:
        d = date.today()

    lunar = _greg_to_lunar(d)
    term_ctx = get_solar_term_context(d)
    season = get_season(d)
    ganzhi = get_tiangan_dizhi(lunar["lunar_year"]) if lunar else ""
    shengxiao = get_shengxiao(d.year)

    lunar_full = ""
    lunar_display = ""
    if lunar:
        lunar_full = f"农历{lunar['month_name']}{lunar['day_name']}"
        lunar_display = f"{ganzhi}年（{shengxiao}） {lunar['month_name']}{lunar['day_name']}"

    return {
        "date": d.isoformat(),
        "weekday": _WEEKDAYS[d.weekday()],
        "lunar_full": lunar_full,
        "lunar_display": lunar_display,
        "lunar_year_name": ganzhi,
        "solar_term_context": term_ctx,
        "season": season,
        "display_date": f"{d.year}年{d.month}月{d.day}日 {_WEEKDAYS[d.weekday()]}",
        "display_lunar": lunar_full,
        "ganzhi_year": f"{ganzhi}年（{shengxiao}年）" if ganzhi else "",
    }


# ============================================================================
# 测试入口
# ============================================================================
if __name__ == "__main__":
    today = date.today()
    ctx = format_date_context(today)
    print("日期上下文:")
    for k, v in ctx.items():
        print(f"  {k}: {v}")
