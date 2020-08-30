import scraping
import scraping_extra
import csv

def main():
    prisoners_array = scraping.scraping_part1()
    print("length before:",len(prisoners_array))
    prisoners_array = scraping_extra.scraping_part2(prisoners_array)
    print("length after:", len(prisoners_array))
    with open('prisoners_dataset.csv', 'w') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')
        for prisoner in prisoners_array:
            wr.writerow(list(prisoner))


if __name__ == '__main__':
    main()