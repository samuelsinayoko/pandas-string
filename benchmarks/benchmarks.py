import pandas as pd
import pandas_string as pd_str
import pyarrow as pa


class TimeSuite:
    def setup(self):
        array = [
            str(x) + str(x) + str(x) if x % 7 == 0 else None
            for x in range(2**15)
        ]
        self.df = pd.DataFrame({'str': array})
        self.df_ext = pd.DataFrame({
            'str': pd_str.StringArray(pa.array(array))
        })

    def time_isnull(self):
        self.df['str'].isnull()

    def time_isnull_ext(self):
        self.df_ext['str'].isnull()

    def time_startswith(self):
        self.df['str'].str.startswith('10')

    def time_startswith_ext(self):
        self.df_ext['str'].text.startswith('10')

    def time_startswith_na(self):
        self.df['str'].str.startswith('10', na=False)

    def time_startswith_na_ext(self):
        self.df_ext['str'].text.startswith('10', na=False)

    def time_endswith_na(self):
        self.df['str'].str.endswith('10', na=False)

    def time_endswith_na_ext(self):
        self.df_ext['str'].text.endswith('10', na=False)
