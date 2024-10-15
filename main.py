from Book_module.Book import Book

def main():
    print("Hello, World!")

    book_one = Book()

    book_1_url = "https://www.gutenberg.org/cache/epub/1695/pg1695.txt"   # The Man Who Was Thursday: A Nightmare
    book_2_url = "https://www.gutenberg.org/cache/epub/70964/pg70964.txt" # The wrong letter
    book_3_url = "https://www.gutenberg.org/cache/epub/1720/pg1720.txt"   # The Man Who Knew Too Much

    book_one.print_info_by_attr("raw_text")

    book_one.get_book(book_1_url)

    # book_one.print_info("raw_text")

if __name__ == "__main__":
    main()