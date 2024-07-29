from sklearn.preprocessing import (
        OneHotEncoder as OE,
        OrdinalEncoder as OrE, 
        StandardScaler as SK_SS,
        MinMaxScaler as SK_MMS,
        )  

class Encoding:

    '''
    This is used to apply encoding on the dataframe.

    Abbreviations:
    ### Encoders:
        'label': LabelEncoder
        'one_hot': OneHotEncoder
    '''

    def __init__(self, _fx, configs):

        self.verbose = configs.get("verbose")

        self.encoder_method = configs.get('encode', {}).get('categorical_encoder_method', 'label')
        self.scaling_method = configs.get('encode', {}).get('numerical_encoder_method', 'standard')

        self.numeric_set = []
        self.categorical_set = []

        self.construct_pipeline_tuples()

    
    def construct_pipeline_tuples(self): 

        # categorical encoding

        if self.encoder_method == "one_hot": 
            ## TODO: add logic to filter out for cols with high cardinality and form feature lists here and then construct it 
            encoding_strategy = OE()
        if self.encoder_method == "label": 
            encoding_strategy = OrE()

        categorical_encoding = ("categorical_encoding", encoding_strategy)
        self.categorical_set.append(categorical_encoding)

        # numerical encoding

        if self.scaling_method == "min_max": 
            encoding_strategy = SK_MMS()
        if self.scaling_method == "standard": 
            encoding_strategy = SK_SS()

        numeric_encoding = ("numeric_encoding", encoding_strategy)
        self.numeric_set.append(numeric_encoding)