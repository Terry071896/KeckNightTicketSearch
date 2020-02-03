import numpy as np
import pandas as pd
import math
from joblib import parallel_backend
import multiprocessing as mp
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import pickle

class SearchTickets(object):
    '''
    A class used as a search algorithm for the Keck Observatory Night Ticket Database

    ...

    Attributes
    ----------
    K : int
        the number of tickets displayed when finally printed
    scale : int
        the number of times important words like Keywords and instruments will be added (if scale is 2, then search with 'kcwi' would be 'kcwi kcwi')
    instruments : list
        the list of insturment names (stem of instrument names)
    stop_words : list
        the list of stop words (the, it, at, be, ...)
    keywords : list
        the list of unique keywords
    words : list
        the processed column Details of the tickets data
    X : matrix
        the sparse matrix of counted words of 'words' from Details column (each column is a word each row is the number of times that word appears in that ticket)
    X_keys : matrix
        the sparse matrix of counted words of 'keywords_orig' from Keyword column (each column is a word each row is the number of times that word appears in that ticket)
    cv : CountVectorizer
        the class used to transform 'words' (Details column) into matrix X
    cv_keys : CountVectorizer
        the class used to transfomr 'keywords_orig' (Keyword column) into matrix X_keys
    df : dataframe
        the ticket data
    model : KMeans
        the KMeans cluster model
    groups : list
        the list of each tickets group according to the KMeans model
    important_tix_index : list
        the list of ticket indecies of the search group
    subset_df : dataframe
        the subset of df where the tickets are associated with the search group only


    Methods
    -------
    fit(df)
        takes ticket dataframe and processes the words and creates attributes: X, cv, words, df, model, and groups
    recluster(k_clusters=14)
        performs k-means to recluster tickets, and then stores model.
    predict(x, x_keys)
        finds the group the search belongs to and ranks the gives a list of points where the higher the points the similar the ticket to the search.
    search(string)
        takes the search and returns sorted ticket list

    _compare_all(x)
        finds the euclidian distance between 2 vectors
    _preprocess_search(search)
        starts threads/parallel to initate string cleaning (_clean_string)
    _clean_string(line)
        takes a string and processes it (removes digits, makes lowercase, ...)
    '''

    def __init__(self, k_tickets, scale = 10):
        '''
        Parameters
        ----------
        k_tickets : int
            number of tickets to display at the end
        scale : int, optional
            number of times that Keywords and instrument words will be scaled (default is 10)
        '''

        self.K = k_tickets # Number of tickets to display at the end
        self.scale = scale # Number of times that Keywords and instruments words will be scaled
        instruments = 'cwi nirspec nirc deimo nire lri mosfir esi hise osiri' # instruments
        self.instruments = instruments.split(' ') # string 'instruments' to list
        stop_words = (stopwords.words('english')) # Import English Stopwords; ex. 'the it be ... '

        SMART = pd.read_csv('SMART_stopwords') # Upload Second (SMART) stopword list as dataframe; this list contains more scientific words
        SMART.columns = ['words'] # Rename the column to 'words'
        for word in SMART.words:  # Loop through words and add new words to 'stop_words'
            if word not in stop_words:
                stop_words.append(word)

        self.stop_words = stop_words

    def fit(self, df):
        '''takes ticket dataframe and processes the words and creates attributes: X, cv, words, df, model, and groups

        Parameters
        ----------
        df : dataframe
            a pandas dataframe of the tickets data.
        '''

        df = df.dropna(subset=['Details', 'Keyword']).reset_index(drop=True) # Drop row if it has an 'NA' in either the Details or Keyword columns
        df.reindex(range(len(df))) # reindex the dataframe

        self.keywords_orig = self._preprocess_search(df.Keyword) # Process the Keywords column

        self.keywords = list(set(self.keywords_orig)) # Create a list of unique processed Keywords

        # words = df.Details + (' '+df.Keyword)*self.scale # create list 'words' and adding it's Keyword as many as the times the scale tells it.
        self.words = self._preprocess_search(df.Details+ (' '+df.Keyword)) # creates list 'words' as processed column Details
        cv = CountVectorizer(binary=False, min_df = 30) # Init 'cv' as CountVectorizer object; Not binary and includes words that appear at least 30 times
        cv_keys = CountVectorizer(binary=False, min_df = 5) # Init 'cv_keys' as CountVectorizer object; Not binary values only and includes words that appear at least 5 times
        cv.fit(self.words) # Fit 'words'. Takes words list and stores the frequency of each word in the entire list for each row
        cv_keys.fit(self.keywords_orig) # Fit 'keywords_orig'. Takes the keywords_orig list and stores the frequency of each word in the entire list of each row
        X = cv.transform(self.words) # Turns 'words' into 'X' matrix where each row is each ticket with counts of words and each column is a unique word that exists in the 'words' list.
        self.X = X.toarray() # Turn 'X' into sparse matrix

        self.cv = cv
        self.cv_keys = cv_keys
        self.df = df

        ###################### CLUSTER MODEL #########################
        # from sklearn.cluster import KMeans
        # self.X_keys = cv_bin.transform(self.keywords)
        # kmeans = KMeans(n_clusters=14, random_state=0).fit(self.X_keys)
        # self.model = kmeans
        # self.groups = kmeans.labels_
        # pickle.dump(kmeans, open('kmeans_14.pkl', 'wb'))
        ##############################################################

        self.model = pickle.load(open('kmeans_14.pkl', 'rb')) # load kmeans model trained with 14 groups
        self.groups = self.model.labels_ # list of what group each ticket belongs too.
        from collections import Counter
        print(Counter(self.groups))

    def recluster(self, k_clusters = 14):
        '''performs k-means to recluster tickets, and then stores model.

        Parameters
        ----------
        k_clusters : int, optional
            the number of clusters wanted to group (default is 14).

        Returns
        -------
        Counter (obj/dict)
            a dictionary in Counter of the number of tickets in each group.
        '''

        from sklearn.cluster import KMeans # imports KMeans from sklearn.cluster
        self.X_keys = self.cv_keys.transform(self.keywords_orig) # Turns 'keywords_orig' into 'X_keys' matrix where each row is each ticket with counts of words and each column is a unique word that exists in the 'keywords_orig' list.
        print(self.X_keys.shape)
        kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(self.X_keys) # clusters the tickets based off of X_keys with k_clusters
        self.model = kmeans # redefines attribute 'model' as kmeans from line above
        self.groups = kmeans.labels_ # redefines attribute 'groups' based off of each tickets respective cluster
        pickle.dump(kmeans, open('kmeans_14.pkl', 'wb')) # restores kmeans model
        from collections import Counter # imports Counter from collections
        return Counter(self.groups)

    def predict(self, x, x_keys):
        '''finds the group the search belongs to and ranks the gives a list of points where the higher the points the similar the ticket to the search.

        Parameters
        ----------
        x : numpy array
            the search broke down by the CountVectorizer associated with the Details column into sparse vector.
        x_keys : numpy array
            the search broke down by the CountVectorizer associated with the Keyword column into sparse vector.

        Returns
        -------
        numpy array
            an array that is associated with each ticket and how closely related it is to the search (the higher the number to closer the ticket)
        '''

        X = self.X # make local 'X' matrix from self.X
        print(X.shape)
        search_group_error = [] # Init search_group_error as a list
        for a in self.model.cluster_centers_: # loop through cluster centers from model (number of groups) store error of each with the search 'x'
            error = np.sqrt(np.sum((a-x_keys)**2))
            search_group_error.append(error)
        search_group_error = np.array(search_group_error) # Make error list a numpy array
        search_group = np.argwhere(search_group_error==np.min(search_group_error))[0][0] # Store min error, which is the group the search would belong too.
        #print(search_group)
        important_tix_index = np.array(np.argwhere(self.groups == search_group).reshape(-1)) # list of index of the tickets of the seared group
        #print(len(important_tix_index))
        important_X_tix = X[important_tix_index] # subset of X matrix with the tickets of the search group

        ######### ERROR METHOD ############# Small error the closest related
        #res = important_X_tix - x # subtract search vector from entire important X rows
        #errors = np.sum(np.abs(res), axis=1) # vector of absolute errors from search
        ####################################

        ######### POINTS METHOD ############## Larger points, the better relation to ticket
        points = np.matmul(important_X_tix, x) # Get points by dot product of earch matrix (important_X_tix) row to search vector (x)
        ######################################

        self.important_tix_index = important_tix_index

        return points #- errors # retrun points or error or points - errors


    def _compare_all(self, x):
        '''finds the euclidian distance between 2 vectors

        Parameters
        ----------
        x : tuple
            a tuple containing search numpy array and row of X matrix

        Returns
        -------
        float
            the euclidian distance between the 2 points (expressed as error)
        '''

        error = np.sqrt(np.sum((x[0]-x[1])**2)) # find euclidian distance as error
        return error


    def _preprocess_search(self, search):
        '''starts threads/parallel to initate string cleaning (_clean_string)

        Parameters
        ----------
        search : str or list
            the search string or list of strings

        Returns
        -------
        new_search : str or list
            the cleaned string or list of cleaned strings
        '''

        if len(search[0]) == 1: # If there is only 1 line in the search, clean the search
            return self._clean_string(search)

        pool = mp.Pool(mp.cpu_count()) # Start number of threads
        new_search = pool.map(self._clean_string, [line for line in search]) # Give tread to each line in search array and store cleaned line
        pool.close() # Kill threads

        return new_search


    def _clean_string(self, line):
        '''takes a string and processes it (removes digits, makes lowercase, ...)

        Parameters
        ----------
        line : str
            a raw line that needs to be cleaned

        Returns
        -------
        str
            the cleaned string
        '''

        REMOVE_NUMBERS = re.compile("[0-9]") # Init Remove any number
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"\(\)\[\]]|[\\n]") # Init replace these characters without a space
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|[+-_]") # Init replace these characters with a space
        ps = PorterStemmer() # Init a PorterStemmer object
        #print(line)
        string = REMOVE_NUMBERS.sub('', line) # Remove numbers
        #print(string)
        string = string.lower() # Make letters lowercase
        string = string.strip() # strip all dead space
        string = string.strip("\n\r\s") # strip newlines and \r and \s
        temp = '' # Init 'temp' string
        string = REPLACE_NO_SPACE.sub('', string) # replace characters in REPLACE_NO_SPACE
        string = REPLACE_WITH_SPACE.sub(" ", string) # replace characters in REPLACE_WITH_SPACE with a space
        #print(string)
        for word in string.split(' '): # Loop through each word
            if word not in self.stop_words: # If not stop word add to string, else nothing
                temp = '%s %s'%(temp, ps.stem(word)) # Add stem of non-stop word.  Stem example: computer and computers become comput

        string = temp # redefine 'string' with no stopword, stem word string
        string = string.strip() # Strip all dead space
        return string


    def search(self, string):
        '''takes the search and returns sorted ticket list

        Parameters
        ----------
        string : str
            the search line

        Returns
        -------
        pandas dataframe
            the sorted tickets of that group (only the first K tickets are returned)
        '''

        words = self._preprocess_search(string) # process 'string'
        temp = '' # Init 'temp' as string
        for word in words.split(' '): # loop over all words in 'words'
            if word not in self.keywords:  # if 'word' is not in 'self.keywords', simply add it, else add 'word' as many times as 'self.scale'
                temp = '%s %s'%(temp, word) # add word to 'temp'
            else:
                temp = '%s%s'%(temp, (' '+word)*self.scale) # add 'word' 'self.scale' times to 'temp'
            if word in self.instruments: # if word is an instrument, add 'word' 'self.scale' times
                temp = '%s%s'%(temp, (' '+word)*self.scale) # add 'word' 'self.scale' times to 'temp'
        words = temp.strip() # strip dead space and redefine 'words'
        #print(words)
        x = self.cv.transform([words]) # transform words into count vector form of X (counting the words in with each associated column in 'words')
        #print(x)
        x = x.toarray() # make sparse vector
        x_keys = self.cv_keys.transform([words])
        x_keys = x_keys.toarray()
        points = self.predict(x[0], x_keys[0]) # predict and return the 'points' (higher the points the better)
        #print(len(chosen), len(self.df), len(self.X))
        self.subset_df = self.df.iloc[self.important_tix_index].reset_index() # Create subset of the group of the tickets of search group
        self.subset_df['points'] = points # add 'points' to the dataframe
        return self.subset_df.sort_values(by=['points', 'Create_date'], ascending=False).head(n=self.K) # return sorted dataframe (top number 'self.K' tickets)
