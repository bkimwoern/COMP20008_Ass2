def filter_out_value(record, column, value):
    """ Filters a DataFrame based on a given column-value pair """
    return record[record[column] == value]
