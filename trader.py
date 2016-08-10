import quandl
import pandas as pd
from sklearn import neighbors, tree, svm, ensemble
from datetime import timedelta, datetime, date
from environment import Environment
import matplotlib.pyplot as plt


class Learner():
    def __init__(self):
        """Dictionary of regressor objects. Corresponding to each symbol - we will have a regressor object"""
        self.regressorDict = dict()

    def train(self, start_date, end_date, symbols):
        """Write code to train here"""

        """Collect data from quandl API"""
        for symbol in symbols :

            quandl.ApiConfig.api_key = "VkbqJssMFzS4q9jpoiaU"

            df = quandl.get("WIKI/{}".format(symbol), start_date = start_date, 
                            end_date = end_date, )

            y_train = df['Adj. Close'].values
            
            X_train = df.drop('Adj. Close', axis=1).values
            
            regressor = tree.DecisionTreeRegressor(max_depth = 8)

            ##regressor = ensemble.RandomForestRegressor(n_estimators = 10, max_depth = 8)

            self.regressorDict[symbol] = regressor
            self.regressorDict[symbol].fit(X_train, y_train)

        print "Training complete"

    def query(self, dates, symbols):
        """Write code to query here"""

        """The query function uses a list of dates and a list of symbols for querying
        The dates would be in the future of the training period dates"""

        """maintain a dataframe that stores the result of each query to observe performance.
           this dataframe would only be used when we are trying to identify the best learner"""

        columns = ['Actual Value', 'Predicted Value', 'Difference']

        df_querydata = pd.DataFrame(index = dates, columns = columns )

        df_querydata.index.names = ['Date']

        for symbol in symbols:
            for date in dates:
                quandl.ApiConfig.api_key = "VkbqJssMFzS4q9jpoiaU"
                
                df = quandl.get("WIKI/{}".format(symbol))

                if date not in df.index:
                    continue
            

                y_test = df.loc[date]

                y_test = y_test['Adj. Close']
                df = df.drop('Adj. Close', axis=1)
                X_test = df.loc[date].values
                
                #Checks whether the symbol is present in the dictionary
                if self.isSymbolPresent(symbol): 
                    y_predict = self.regressorDict[symbol].predict(X_test.reshape(1,-1)) ##passing 1-d array as data deprecated

                    #print("Actual value of adjusted close is {} and predicted value is {} on the day {}".format(y_test, y_predict, date))

                    df_querydata.loc[date]['Actual Value'] = y_test
                    df_querydata.loc[date]['Predicted Value'] = y_predict[0]
                    df_querydata.loc[date]['Difference'] = (abs(y_test-y_predict[0])/(y_test)) * 100

                else:
                    print("Wrong value of symbol. Please add a symbol that has been trained")

            print "Results for {}".format(symbol)

            print df_querydata

            mape = df_querydata['Difference'].mean() #calculate mean absolute percenatge error

            print mape

            plotFigure(df_querydata)

        

    
    def isSymbolPresent(self, symbol):

        """this function checks if a symbol is already present in the dictionary as a key"""

        for symbols in self.regressorDict.keys():
            if symbol == symbols:
                return True

        return False

def plotFigure(df):
    ax = df.plot(y='Actual Value', label='Actual Value')
    df.plot(y='Predicted Value', label = 'Predicted Value', ax= ax)

    ax.legend(loc='upper right')

    plt.show()

def validate(date):
    try:
        datetime.strptime(str(date),'%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect date format - should be YYYY-MM-DD")

def isSubset(query_list, training_list):
    if (set(query_list) <= set(training_list)):
        return True
    else:
        return False


def weekDay(date):
    date = datetime.strptime(str(date), '%Y-%m-%d')

    return date.weekday()

def trainAndQuery():
    """This function will create the learner object and perform the training and
    query functions"""

    learner = Learner()

    print("Start entering details of the start date for training")

    print("Enter the start date in YYYY-MM-DD format. Use '' to input date.")
    start_date = str(input('start-date: '))

    validate(start_date)

    print("Enter the end date in YYYY-MM-DD format. Use '' to input date.")


    end_date = str(input('end-date: '))

    validate(end_date)
    print("Enter stock symbols, separated by commas(do not include spaces in between),"\
          "you are going to use for training and prediction")

    symbols = str(raw_input('symbols: '))
        
    symbols_list = symbols.split(",")

    symbols_list = list(set(symbols_list)) ##remove duplicates

    learner.train(start_date, end_date, symbols_list)

    """Give user two options- either pass the query date yourself- or we calculate it"""

    query_dates = []


    options = str(raw_input("Do you want to pass the query dates yourself(Write Yes or No): "))

    if (options.lower()=='yes'):
        dates = str(input('query-dates(in YYYY-MM-DD format) separated by commas, no spaces allowed: '))

        query_dates = dates.split(",")
        query_dates = list(set(query_dates)) ##remove duplicates

        for i, date in enumerate(query_dates):
            
            validate(str(date))
            day_number = weekDay(date)
            if(day_number == 5):
                raise ValueError("The date corresponds to a saturday. Please enter a day between Monday-Friday")
                                 
            elif(day_number == 6):
                raise ValueError("The date corresponds to a sunday. Please enter a day between Monday-Friday")

            
    elif (options.lower()=='no'):
        
        ## check if the end-date is a weekend ##
        day_number = weekDay(end_date)
        if(day_number == 5):
            raise ValueError("The end of the training date is a Saturday. Please change it to a day between Monday- Friday")

        elif(day_number == 6):
            raise ValueError("The end of the training date is a Saturday. Please change it to a day between Monday- Friday")

        
        days = 7
        start = pd.to_datetime(end_date)
        while ((start + timedelta(days)) < pd.to_datetime('2011-12-31')):

            date_to_add = start + timedelta(days)
            query_dates.append(date_to_add)
            start = pd.to_datetime(date_to_add)
            

    else:
        print ("Wrong input passed !!!")
        
        
    print("Now pass the stock symbols for query, separated by commas, without any spaces")
    
    query_symbols = str(raw_input("query-symbols: "))

    query_symbols_list = query_symbols.split(",")
    query_symbols_list = list(set(query_symbols_list)) ##remove duplicates

    """Check if the query symbols list is a subset of the training symbols list"""

    
    if isSubset(query_symbols_list, symbols_list):
        learner.query(query_dates, query_symbols_list)
    else:
        print ("Choose the query symbols that have been trained already!!")


def suggestActions():
    """Predicts action to be taken on stocks"""
    print("Please type the symbol you want to train-- ")

    symbol = str(raw_input("symbol: "))

    print("Enter the start dates and end dates for training-- these dates must be atleast 2000 days (more than 5 years) away."\
           "The dates must be in 'YYYY-MM-DD' format")

    training_start_date = str(input("training start_date: "))

    training_end_date = str(input("training end_date: "))

    status = validateTwoDates(training_start_date, training_end_date)

    testingState = False
    if status:
        env = Environment()

        df = env.collectData(symbol, training_start_date, training_end_date)

        discreteStates, daily_returns = env.prepareStates(df)

        for i in range(0, discreteStates.shape[0] - 1):
            reward = env.update(discreteStates.iloc[i : i+2], daily_returns.iloc[i], discreteStates.index[i], testingState, i+1)
            

    else:
        #make sure the training period is large enough -- I have used a 2000 day period
        raise ValueError("Either the date format is wrong. Or the difference between the start date"\
                          "and end-date is less than 2000 days.")

    print("Training is done. Get ready to test now.")

    testingState = True

    print("Enter the start dates and end dates for testing--the dates must be in 'YYYY-MM-DD' format")

    ##Test date must be later than training date--
    testing_start_date = str(input("testing start_date: "))
    validate(testing_start_date)

    if(testing_start_date <= training_end_date):
        raise ValueError("Please enter a testing date later than the training period.")

    testing_start_date = pd.to_datetime(testing_start_date)

    testing_end_date = str(input("testing end_date: "))
    validate(testing_end_date)

    df = env.collectData(symbol, testing_start_date, testing_end_date)

    discreteStates, daily_returns = env.prepareStates(df)


    sumOfRewards = 0
    for i in range(0, discreteStates.shape[0]-1):
        reward = env.update(discreteStates.iloc[i : i+2], daily_returns.iloc[i], discreteStates.index[i], testingState, i+1)
        sumOfRewards = sumOfRewards + reward

    print "Net reward is {}".format(sumOfRewards)

    

def validateTwoDates(start_date, end_date):
    try:
        a = datetime.strptime(str(start_date),'%Y-%m-%d')
        b = datetime.strptime(str(end_date),'%Y-%m-%d')

        diff = b - a

        if (diff.days < 2000):
            return False
        else:
            return True
    except:
        return False



def run():
    """Interactive Menu based for user to determine what to do"""
    options = {
                1 : trainAndQuery,
                2 : suggestActions
                }
    
    print("What do you want to do? Choose one from below :-")
    print("1 - Train and Query Stocks based on various dates")
    print("2 - Suggest good stocks to buy/sell at certain periods of time")

    choice = int(input('What do you choose- Type the number corresponding to your choice'\
                        ' of action : '))

    options[choice]()

    
    
if __name__ == '__main__':
    run()









                                   










