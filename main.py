from Book_module.Book import Book

def main():

    book_one = Book(book_number = 1)
    book_two = Book(book_number = 2)
    book_three = Book(book_number = 3)

    book_1_url = "https://www.gutenberg.org/cache/epub/1695/pg1695.txt"   # The Man Who Was Thursday: A Nightmare
    book_2_url = "https://www.gutenberg.org/cache/epub/204/pg204.txt" # The innonce of father brown
    book_3_url = "https://www.gutenberg.org/cache/epub/1720/pg1720.txt"   # The Man Who Knew Too Much

    book_one.get_book(book_1_url, from_txt=True, txt_file_path="books/book_one.txt")
    book_two.get_book(book_2_url, from_txt=True, txt_file_path="books/book_two.txt")
    book_three.get_book(book_3_url, from_txt=True , txt_file_path="books/book_three.txt")
    # book_three.pre_process()
    book_one.pre_process()
    # book_three.print_info_by_attr("names")
    # book_three.print_info_by_attr("special_characters")
    book_one.tokenize()

if __name__ == "__main__":
    main()