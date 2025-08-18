import streamlit as st

st_s = st.session_state
def set_session_states():
    if "t10_button" not in st_s:
        st_s.t10_button = False
    if "book_button" not in st_s:
        st_s.book_button = False
    if "genres" not in st_s:
        st_s.genres = []
    if "selected_book" not in st_s:
        st_s.selected_book = None
    if "actual_title" not in st_s:
        st_s.actual_title = None


