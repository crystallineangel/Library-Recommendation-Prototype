import streamlit as st
import pandas as pd

st.set_page_config(page_title='Bookmarked', page_icon="📚", layout="wide")

# basic setup
st_s = st.session_state
df = pd.read_csv("complete_data.csv")

import prototype_session_state as ss
import prototype_rec_system as rec

ss.set_session_states()

st.title(":books: _Bookmarked_ ")

col1, col2, col3 = st.columns((4, 2, 7), gap="small")


with col1:
    st.header("Top books from each genre")
    # Way too many sub-genres/categories, would be overwhelming to just have everything listed out

    st.write("No books in mind and only genres/subjects? That's ok, tell us your preferred genres and we'll find the top books!")
    st_s.genres = st.multiselect(
        "Choose your subjects:",
        rec.unique_genres(df)
    )

    if st.button("Get the top books"):
        st_s.t10_button = True

    if st_s.t10_button and st_s.genres:
        top_books = rec.get_top_books_by_genre(df, st_s.genres)

        # Technical coding by ChatGPT
        for genre, books in top_books.items():
            with st.expander(f"Top Books in {genre}"):
                for _, row in books.iterrows():
                    st.write(f"**{row['title']}** by *{row['author']}*")
                    st.write(f"⭐ Rating: {row['rating']}")
                    st.write(f"📖 {row['descr']}...")
                    st.markdown("---")
    elif st_s.t10_button:
        st.error("Please select at least one genre to get recommendations!")

with col2:
    st.image("bm.jpg", use_container_width=True)

with col3:
    st.header("Just like your faves")
    st.write("Enter your favorite books and we'll find the similar ones, straight from our library")

    st_s.selected_book = st.selectbox(
        "Choose a book:",
        rec.get_book_list(df),
        index=None
    )

    # display selected book
    st.write(f"You selected: {st_s.selected_book}. Click the following button to get similar recommendations!")

    # Normalize and clean the selected title (remove the author part and the '--' separator)

    if st.button("Get my recs!"):
        st_s.book_button = True

    if st_s.book_button and st_s.selected_book is not None:
        st_s.actual_title = st_s.selected_book.split(' -- ')[0]

        recommended_books = rec.get_recs(df, st_s.actual_title)

        if not recommended_books.empty:
            st.divider()
            st.write(f"Hi, here are your recommendations!")

            if rec.not_enough_recs = True:
                st.write(f"Only found {len(rec.final_recs)} books instead of {len(rec.rec_num)} in our current library!")

            for idx, row in recommended_books.iterrows():
                name = str(row['title'])
                author = str(row['author'])
                label = name + " -- " + author
                with st.popover(label):
                    rating = rec.get_rating(df, row['title'])
                    descr = rec.get_descr(df, row['title'])

                    st.write(f"**{row['title']}** by *{row['author']}*")
                    st.write(f"⭐ Rating: {rating}")
                    st.write(f"📖 Description: {descr}")
                    st.markdown("---")

    elif st_s.book_button:
        st.error("Hey there, be sure to input your book before getting the rec!")







