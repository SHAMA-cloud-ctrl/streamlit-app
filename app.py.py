import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# تخصيص الألوان
st.markdown("<h1 style='color:#1E90FF;'>محاكي التحسين المالي</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#32CD32;'>هنا يمكنك تحسين محفظتك الاستثمارية بناءً على القيود التنظيمية.</p>", unsafe_allow_html=True)

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

# بيانات ثابتة للعوائد والأصول
mu = np.array([0.1, 0.15, 0.12, 0.08, 0.14])  # العوائد المتوقعة للأصول
Sigma = np.array([[0.1, 0.02, 0.03, 0.01, 0.02],
                  [0.02, 0.12, 0.04, 0.02, 0.03],
                  [0.03, 0.04, 0.15, 0.02, 0.05],
                  [0.01, 0.02, 0.02, 0.1, 0.01],
                  [0.02, 0.03, 0.05, 0.01, 0.11]])  # مصفوفة التباين والتغاير للأصول
n = len(mu)  # عدد الأصول

# شريط تمرير لتحديد المخاطرة
risk_limit = st.sidebar.slider("🔒 الحد الأقصى للمخاطرة", 0.0, 0.1, 0.05)

# قائمة متعددة لاختيار الأصول غير المتوافقة مع الشريعة
non_halal = st.sidebar.multiselect("🕌 الشركات غير المتوافقة مع الشريعة", options=list(range(n)))

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
