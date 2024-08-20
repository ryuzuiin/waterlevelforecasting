class CSVColumnExtractor:
    def __init__(self, directory, file_column_map, time_column):
        """
        Initialize with the directory of CSV files, a mapping of filenames to desired columns, and the time column.
        :param directory: Directory where CSV files are located.
        :param file_column_map: Dictionary mapping filenames to lists of desired columns.
        :param time_column: Name of the common time column to sort by.
        """
        self.directory = directory
        self.file_column_map = file_column_map
        self.time_column = time_column

    def extract_columns(self):
        """
        Extract the specified columns from each CSV file, convert time column to datetime, sort by the time column,
        and combine them into a single DataFrame.
        :return: Combined DataFrame with the selected columns, sorted by the time column.
        """
        data = {}

        for file, columns in self.file_column_map.items():
            file_path = os.path.join(self.directory, file)
            try:
                df = pd.read_csv(file_path)
                # Ensure the time column is included in the selection
                selected_columns = df[[self.time_column] + columns].copy()
                # Convert the time column to datetime format
                selected_columns.loc[:, self.time_column] = pd.to_datetime(selected_columns[self.time_column])
                # Sort the DataFrame by the time column
                selected_columns = selected_columns.sort_values(by=self.time_column)
                # Ensure the time column is unique
                selected_columns = selected_columns.drop_duplicates(subset=[self.time_column])
                # Set the time column as the index for proper alignment
                data[file] = selected_columns.set_index(self.time_column)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # Concatenate the selected columns from each file horizontally
        combined_df = pd.concat(data.values(), axis=1)

        return combined_df.reset_index()