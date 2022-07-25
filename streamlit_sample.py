import predict_modify as pm
import streamlit as st

# import streamlit as st

pm.init_logger()

st.title("KoBERT NER Test")
text = st.text_area("텍스트 입력 : ")
masking_ner = st.text_area("마스킹 할 객체 입력(띄어쓰기로 구분) : ")

if st.button("apply"):
    masking_ner = masking_ner.split(" ")
    result = pm.predict(text, masking_ner)
    con = st.container()
    con.caption("적용 결과")
    con.write(result)
    print(result)

# masking_ner = st.text_area("마스킹 할 객체 입력(띄어쓰기로 구분) : ")

# if st.button("Masking"):
#     masking_ner = masking_ner.split(" ")

#     con = st.container()
#     con.caption("검출됟 객체에 대한 마스킹 적용 결과")
#     con.write(pm.predict(text, masking_ner))

# output = pm.predict(
#     """유행기 6승 29패 .
# 강변관광지 '보르도'에 취하다
# 특별한인연, R&D 신문로 증수
# 원정에서는 구단이 만든 상대자 원료를 본다 ."""
# )

# masking_ner = ["NUM-B", "AFW-B"]

# output_masked = pm.predict(
#     """유행기 6승 29패 .
# 강변관광지 '보르도'에 취하다
# 특별한인연, R&D 신문로 증수
# 원정에서는 구단이 만든 상대자 원료를 본다 .""",
#     masking_ner,
# )

# print(output)

# print(output_masked)
