import streamlit as st


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "rapt-AI";
                margin-left: 20px;
                margin-top: 0px;
                font-size: 50px;
                position: relative;
                top: 0px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


pages = {
    "Process": [
        st.Page("process.py", title="Upload Documents"),

    ],
    "Chat": [
        st.Page("chat.py", title="Chat Bot"),
    ],
}

st.set_page_config(page_title="rapt-AI", layout="wide")
pg = st.navigation(pages)
add_logo()
pg.run()
