import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="Định giá Bất động sản Hà Nội AI",
    page_icon="🏠",
    layout="wide"
)

# Tinh chỉnh khoảng cách lề trên và style thẻ kết quả
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .header-banner {
        background: linear-gradient(135deg, #0f2942 0%, #1e4e79 100%);
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(15, 41, 66, 0.12);
    }
    .header-banner h2 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
    }
    .header-banner p {
        margin: 0.35rem 0 0 0;
        font-size: 0.92rem;
        color: #d9e6f2;
    }
    .price-highlight {
        background-color: #f0f7ff;
        border-left: 5px solid #1e4e79;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.98rem;
        color: #0f2942;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    df_clean = pd.read_csv("Data/processed/hanoi_housing_cleaned.csv")
    artifact = joblib.load("models/best_housing_model.pkl")
    return df_clean, artifact

df, artifact = load_resources()

# Banner tiêu đề gọn đẹp, không bị khuất mép trên
st.markdown("""
<div class="header-banner">
    <h2>🏠 Hệ thống Định giá Bất động sản Hà Nội</h2>
    <p>Ứng dụng mô hình CatBoost Regressor • Huấn luyện trên 7.365 bản ghi đã chuẩn hóa đa nguồn</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.05, 1.15], gap="large")

# ==============================================================================
# CỘT TRÁI: KHUNG NHẬP THÔNG SỐ
# ==============================================================================
with col_left:
    with st.container(border=True):
        st.subheader("1. Thông số bất động sản")

        c1, c2 = st.columns(2)
        with c1:
            districts = sorted(df["Khu vực/Quận huyện theo nguồn"].dropna().unique().tolist())
            default_idx = districts.index("Cầu Giấy") if "Cầu Giấy" in districts else 0
            district = st.selectbox("Quận / Huyện (*)", districts, index=default_idx)

            house_types = sorted(df["Loại nhà"].dropna().unique().tolist())
            house_type = st.selectbox("Loại hình nhà ở (*)", house_types)

        with c2:
            wards_in_dist = sorted(
                df[df["Khu vực/Quận huyện theo nguồn"] == district]["Phường xã"].dropna().unique().tolist()
            )
            ward_options = ["Không rõ"] + [w for w in wards_in_dist if w != "Không rõ"]
            ward = st.selectbox("Phường / Xã", ward_options)

            projects = sorted(df["Tên dự án nếu có"].dropna().unique().tolist())
            project_options = ["Không rõ"] + [p for p in projects if p != "Không rõ"]
            project_name = st.selectbox("Tên dự án (nếu có)", project_options)

        st.divider()

        c3, c4 = st.columns(2)
        with c3:
            area = st.number_input("Diện tích (m²) (*)", min_value=10.0, max_value=1000.0, value=60.0, step=1.0)
            bedrooms = st.number_input("Số phòng ngủ", min_value=1, max_value=20, value=3, step=1)
        with c4:
            floors = st.number_input("Số tầng", min_value=1, max_value=15, value=4, step=1)
            bathrooms = st.number_input("Số phòng vệ sinh", min_value=1, max_value=20, value=3, step=1)

        st.write("")
        predict_btn = st.button("🔍 Định giá bất động sản", type="primary", use_container_width=True)

# ==============================================================================
# CỘT PHẢI: KHUNG KẾT QUẢ & THAM CHIẾU THỊ TRƯỜNG
# ==============================================================================
with col_right:
    with st.container(border=True):
        st.subheader("2. Kết quả định giá & Tham chiếu")

        # Chạy dự đoán mặc định ngay khi mở trang để giao diện luôn đầy đặn
        input_df = pd.DataFrame([{
            "Khu vực/Quận huyện theo nguồn": district,
            "Phường xã": ward,
            "Diện tích": float(area),
            "Loại nhà": house_type,
            "Số tầng": float(floors),
            "Số phòng ngủ": float(bedrooms),
            "Số phòng vệ sinh": float(bathrooms),
            "Tên dự án nếu có": project_name,
        }])

        if isinstance(artifact, dict) and "model" in artifact:
            for col in artifact["num_cols"]:
                input_df[col] = input_df[col].fillna(artifact["medians"].get(col, 0))
            for col in artifact["cat_cols"]:
                input_df[col] = input_df[col].fillna("Không rõ").astype(str)
            pred_vnd = float(artifact["model"].predict(input_df)[0])
        else:
            pred_vnd = float(artifact.predict(input_df)[0])

        pred_vnd = max(pred_vnd, 1e8)
        pred_billion = pred_vnd / 1e9
        unit_price_mil = (pred_vnd / area) / 1e6

        # Tính chênh lệch so với trung vị cùng Quận & Loại nhà
        sub_ref = df[
            (df["Khu vực/Quận huyện theo nguồn"] == district) &
            (df["Loại nhà"] == house_type)
        ]
        delta_text = None
        if len(sub_ref) >= 3:
            med_unit = ((sub_ref["Giá chuẩn hóa VNĐ"] / sub_ref["Diện tích"]) / 1e6).median()
            diff_pct = ((unit_price_mil - med_unit) / med_unit) * 100
            delta_text = f"{diff_pct:+.1f}% so với trung vị {house_type}"

        m1, m2 = st.columns(2)
        m1.metric("Tổng giá ước tính", f"{pred_billion:,.2f} tỷ VNĐ")
        m2.metric("Đơn giá ước tính", f"{unit_price_mil:,.1f} triệu/m²", delta=delta_text, delta_color="off")

        st.markdown(
            f'<div class="price-highlight">💰 <b>Giá quy đổi chính xác:</b> {pred_vnd:,.0f} VNĐ</div>',
            unsafe_allow_html=True
        )

        # Bảng & Biểu đồ mặt bằng giá tại Quận đang chọn
        dist_df = df[df["Khu vực/Quận huyện theo nguồn"] == district].copy()
        if not dist_df.empty:
            dist_df["Đơn giá (Triệu/m²)"] = (dist_df["Giá chuẩn hóa VNĐ"] / dist_df["Diện tích"]) / 1e6
            summary = (
                dist_df.groupby("Loại nhà")
                .agg(
                    **{
                        "Loại hình nhà ở": ("Loại nhà", "first"),
                        "Số tin": ("Mã bản ghi", "count"),
                        "Giá trung vị (Tỷ)": ("Giá chuẩn hóa VNĐ", lambda s: round(s.median() / 1e9, 2)),
                        "Đơn giá trung vị (Tr/m²)": ("Đơn giá (Triệu/m²)", lambda s: round(s.median(), 1)),
                    }
                )
                .reset_index(drop=True)
                .sort_values("Đơn giá trung vị (Tr/m²)", ascending=False)
            )

            # Biểu đồ cột ngang so sánh đơn giá ước tính vs mặt bằng Quận
            chart_df = summary[["Loại hình nhà ở", "Đơn giá trung vị (Tr/m²)"]].copy()
            chart_df.columns = ["Phân loại", "Đơn giá (Triệu/m²)"]
            chart_df["Nhóm"] = "Trung vị khu vực"

            pred_row = pd.DataFrame([{
                "Phân loại": f"⭐ Căn đang định giá ({house_type})",
                "Đơn giá (Triệu/m²)": round(unit_price_mil, 1),
                "Nhóm": "Ước tính AI"
            }])
            chart_df = pd.concat([pred_row, chart_df], ignore_index=True)

            st.markdown(f"**So sánh đơn giá tại {district} (Triệu VNĐ/m²):**")
            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusEnd=5, height=22)
                .encode(
                    x=alt.X("Đơn giá (Triệu/m²):Q", title="Đơn giá (Triệu VNĐ/m²)"),
                    y=alt.Y("Phân loại:N", sort="-x", title=""),
                    color=alt.Color(
                        "Nhóm:N",
                        scale=alt.Scale(
                            domain=["Ước tính AI", "Trung vị khu vực"],
                            range=["#e65100", "#1e4e79"]
                        ),
                        legend=None
                    ),
                    tooltip=["Phân loại", "Đơn giá (Triệu/m²)"]
                )
                .properties(height=185)
            )
            st.altair_chart(bar_chart, use_container_width=True)

            with st.expander(f"📋 Xem bảng số liệu chi tiết tại {district}", expanded=True):
                st.dataframe(summary, use_container_width=True, hide_index=True)