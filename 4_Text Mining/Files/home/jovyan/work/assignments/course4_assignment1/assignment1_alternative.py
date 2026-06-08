import pandas as pd

doc = []
with open(
    "C:\\Users\\singl\\PYTHON\\Univ of Michigan\\Applied Data Science -University of Michigan\\Text Mining\\Files\\home\\jovyan\\work\\assignments\\course4_assignment1\\assets\\dates.txt"
) as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)


def date_sorter():

    # Your code here
    # Full date

    # extract dates from original dataframe and saved into 3 new cols, incl a) any digit of 'days' b) any digit of 'month' and c) any digit with 4 repetitons of 'year'
    # e.g., 04/20/2009
    messay_date = df.str.extractall(
        r"(?P<origin>(?P<month>\d?\d)[/|-](?P<day>\d?\d)[/|-](?P<year>\d{4}))"
    )
    messay_index = ~df.index.isin([x[0] for x in messay_date.index])

    # extract and append a) any digit of 'month' b) days e.g., 11/29 and c) two digit of 'year' e.g., 89/90
    # e.g., 6/2008; 12/2009
    messay_date = messay_date.append(
        df[messay_index].str.extractall(
            r"(?P<origin>(?P<month>\d?\d)[/|-](?P<day>([0-2]?[0-9])|([3][01]))[/|-](?P<year>\d{2}))"
        )
    )
    messay_index = ~df.index.isin([x[0] for x in messay_date.index])

    # extract and append a) any digit of 'day' b) at least 3 characters of 'month' e.g., JAN/feb and b) any digit with 4 repetitons of 'year'
    # e.g., 20 Mar 2009
    messay_date = messay_date.append(
        df[messay_index].str.extractall(
            r"(?P<origin>(?P<day>\d?\d) ?(?P<month>[a-zA-Z]{3,})\.?,? (?P<year>\d{4}))"
        )
    )
    messay_index = ~df.index.isin([x[0] for x in messay_date.index])

    # extract and append a) at least 3 characters of 'month' e.g., JAN/feb b) any dight of 'day' plus 'th/nd/st' and c) any digit with 4 repetitons of 'year'
    # Mar 20th, 2009
    messay_date = messay_date.append(
        df[messay_index].str.extractall(
            r"(?P<origin>(?P<month>[a-zA-Z]{3,})\.?-? ?(?P<day>\d\d?)(th|nd|st)?,?-? ?(?P<year>\d{4}))"
        )
    )
    messay_index = ~df.index.isin([x[0] for x in messay_date.index])

    del messay_date[3]
    del messay_date[4]

    # if the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009)
    dates_without_day = df[messay_index].str.extractall(
        "(?P<origin>(?P<month>[A-Z][a-z]{2,}),?\.? (?P<year>\d{4}))"
    )
    dates_without_day = dates_without_day.append(
        df[messay_index].str.extractall(r"(?P<origin>(?P<month>\d\d?)/(?P<year>\d{4}))")
    )
    dates_without_day["day"] = 1
    messay_date = messay_date.append(dates_without_day)
    messay_index = ~df.index.isin([x[0] for x in messay_date.index])

    # Only year
    dates_only_year = df[messay_index].str.extractall(r"(?P<origin>(?P<year>\d{4}))")
    dates_only_year["day"] = 1
    dates_only_year["month"] = 1
    messay_date = messay_date.append(dates_only_year)
    messay_index = ~df.index.isin([x[0] for x in messay_date.index])

    # normalized year
    messay_date["year"] = messay_date["year"].apply(
        lambda x: "19" + x if len(x) == 2 else x
    )
    messay_date["year"] = messay_date["year"].apply(lambda x: str(x))

    # normalized month
    messay_date["month"] = messay_date["month"].apply(
        lambda x: x[1:] if type(x) is str and x.startswith("0") else x
    )
    month_dict = dict(
        {
            "September": 9,
            "Mar": 3,
            "November": 11,
            "Jul": 7,
            "January": 1,
            "December": 12,
            "Feb": 2,
            "May": 5,
            "Aug": 8,
            "Jun": 6,
            "Sep": 9,
            "Oct": 10,
            "June": 6,
            "March": 3,
            "February": 2,
            "Dec": 12,
            "Apr": 4,
            "Jan": 1,
            "Janaury": 1,
            "August": 8,
            "October": 10,
            "July": 7,
            "Since": 1,
            "Nov": 11,
            "April": 4,
            "Decemeber": 12,
            "Age": 8,
        }
    )
    messay_date.replace({"month": month_dict}, inplace=True)
    messay_date["month"] = messay_date["month"].apply(lambda x: str(x))

    # Day
    messay_date["day"] = messay_date["day"].apply(lambda x: str(x))

    # Cleaned date
    messay_date["date"] = (
        messay_date["month"] + "/" + messay_date["day"] + "/" + messay_date["year"]
    )
    messay_date["date"] = pd.to_datetime(messay_date["date"])

    messay_date.sort_values(by="date", inplace=True)
    df1 = pd.Series(list([x[0] for x in messay_date.index]))

    return df1  # Your answer here


df_answer = date_sorter()
