import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# تخصيص الألوان
st.markdown("<h1 style='color:#1E90FF;'>محاكي التحسين المالي</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#32CD32;'>هنا يمكنك تحسين محفظتك الاستثمارية بناءً على القيود التنظيمية والمالية.</p>", unsafe_allow_html=True)

# تخصيص الأزرار باستخدام CSS
st.markdown("""
    <style>
        .stButton > button {
            background-color: #1E90FF;
            color: white;
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #4682B4;
        }
    </style>
""", unsafe_allow_html=True)

# تحميل ملف البيانات
uploaded_file = st.file_uploader("📁 حمّل ملف CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # التحقق من الأعمدة في البيانات
    if "returns" not in df.columns:
        st.error("❌ لا يحتوي الملف على عمود 'returns'.")
    else:
        # تخصيص الأعمدة تلقائيًا
        mu = df["returns"].values
        Sigma = df.drop("returns", axis=1).values  # باقي الأعمدة هي التغاير
        n = len(mu)

        # شريط تمرير لتحديد المخاطرة
        risk_limit = st.sidebar.slider("🔒 الحد الأقصى للمخاطرة", 0.0, 0.1, 0.05)

        # قائمة متعددة لاختيار الأصول غير المتوافقة مع الشريعة
        non_halal = st.sidebar.multiselect("🕌 الشركات غير المتوافقة مع الشريعة", options=list(range(n)))

        # تخصيص القيود المالية: الحد الأدنى والأقصى للاستثمار في الأصول
        min_allocation = st.sidebar.slider("📉 الحد الأدنى للاستثمار في الأصل (%)", 0.0, 0.2, 0.05)
        max_allocation = st.sidebar.slider("📈 الحد الأقصى للاستثمار في الأصل (%)", 0.2, 1.0, 0.3)

        # زر لتشغيل التحسين
        if st.button('🎯 تحسين المحفظة'):
            # إعداد نموذج التحسين
            w = cp.Variable(n)
            risk = cp.quad_form(w, Sigma)
            ret = mu @ w

            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                risk <= risk_limit
            ]

            for i in non_halal:
                constraints.append(w[i] == 0)
            
            for i in range(n):
                constraints.append(w[i] >= min_allocation)
                constraints.append(w[i] <= max_allocation)

            prob = cp.Problem(cp.Maximize(ret), constraints)
            prob.solve()

            st.markdown("<h2 style='color:green;'>نتائج المحفظة المثلى</h2>", unsafe_allow_html=True)

            if w.value is not None:
                st.success("✅ تم حساب المحفظة المثلى بنجاح!")

                # تخصيص الرسم البياني
                fig, ax = plt.subplots()
                ax.bar(range(n), w.value, color=['#FF6347', '#1E90FF', '#32CD32', '#FFD700', '#8A2BE2'])  # ألوان مخصصة
                ax.set_title('أوزان المحفظة الاستثمارية', fontsize=14, color='darkblue')
                ax.set_xlabel('الأصول', fontsize=12, color='navy')
                ax.set_ylabel('النسبة', fontsize=12, color='navy')

                # عرض الرسم البياني في Streamlit
                st.pyplot(fig)
            else:
                st.error("❌ لم يتم العثور على حل ضمن القيود المدخلة.")
else:
    st.info("يرجى تحميل ملف CSV لبدء التحليل.")
