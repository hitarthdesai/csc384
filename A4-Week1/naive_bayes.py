############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 4 Starter Code
## v1.0
############################################################

from collections import Counter
from bnetbase import Variable, Factor, BN
import csv

def convert_factor_table_key_to_tuple(factor_table):
    new_factor_table = {}
    for key, prob in factor_table.items():
        components = filter(lambda x: len(x.strip()) > 0, key.split(','))
        new_key = []
        for component in components:
            first_equal = component.index("=")
            value = component[first_equal+1:].strip()
            new_key.append(value)
        new_key = tuple(new_key)
        new_factor_table[new_key] = prob

    return new_factor_table

def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''

    new_factor_name = f"Normalized_{factor.name}"
    new_factor = Factor(new_factor_name, factor.scope)

    factor_table = convert_factor_table_key_to_tuple(factor.get_table())
    total = sum(factor_table.values())
    if total == 0:
        return factor
    new_factor_values = list(map(lambda i: list(i[0]) + [i[1]/total], factor_table.items()))
    new_factor.add_values(new_factor_values)

    return new_factor


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    '''

    var_index = next(i for i, v in enumerate(factor.scope) if v.name == variable.name)
    new_factor_name = f"Restricted_{factor.name}_by_{variable}={value}"
    new_factor_scope = list(filter(lambda v: v.name != variable.name, factor.scope))
    new_factor = Factor(new_factor_name, new_factor_scope)

    new_factor_values = []
    factor_table = convert_factor_table_key_to_tuple(factor.get_table())
    for key, prob in factor_table.items():
        if key[var_index] != value:
            continue

        t = [None] * len(new_factor_scope)
        for i, val in enumerate(key):
            if i < var_index:
                t[i] = val
            elif i > var_index:
                t[i-1] = val

        t.append(prob)
        new_factor_values.append(t)

    new_factor.add_values(new_factor_values)
    return new_factor


def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''

    var_index = list(i for i, v in enumerate(factor.scope) if v.name == variable.name)
    if len(var_index) == 0:
        return factor
    var_index = var_index[0]

    new_factor_name = f"Summed_out_{factor.name}_by_{variable.name}"
    new_factor_scope = list(filter(lambda v: v.name != variable.name, factor.scope))
    new_factor = Factor(new_factor_name, new_factor_scope)

    summed_out_table = {}
    factor_table = convert_factor_table_key_to_tuple(factor.get_table())
    for key, prob in factor_table.items():
        new_key = list(key)
        new_key.pop(var_index)
        new_key = tuple(new_key)
        summed_out_table[new_key] = summed_out_table.get(new_key, 0) + prob

    values = list(map(lambda i: list(i[0]) + [i[1]], summed_out_table.items()))
    new_factor.add_values(values)
    return new_factor



def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''

    if len(factor_list) == 0:
        return Factor("Empty_Factor", [])

    new_factor_name = f"Multiplied_{'_'.join([f.name for f in factor_list])}"

    new_factor_scope = set()
    for factor in factor_list:
        new_factor_scope.update(factor.scope)
    new_factor_scope = list(new_factor_scope)
    new_factor = Factor(new_factor_name, new_factor_scope)

    common_vars = set()
    for factor in factor_list:
        common_vars.intersection_update(factor.scope)
    common_vars = list(common_vars)


    first_factor = factor_list[0]
    first_factor_table = convert_factor_table_key_to_tuple(first_factor.get_table())
    for key, prob in first_factor_table.items():

        new_values = [None] * len(new_factor_scope)
        stuff = zip(first_factor.scope, key)
        for var, val in stuff:
            idx = next((i for i, v in enumerate(new_factor_scope) if v.name == var.name), None)
            new_values[idx] = val

        for second_factor in factor_list[1:]:
            second_factor_table = convert_factor_table_key_to_tuple(second_factor.get_table())

            for second_key, second_prob in second_factor_table.items():
                same_common_vars = all(key[i] == second_key[i] for i in range(len(key)))
                if not same_common_vars:
                    continue
                
                stuff = zip(first_factor.scope, key)
                for var, val in stuff:
                    idx = next((i for i, v in enumerate(new_factor_scope) if v.name == var.name), None)
                    new_values[idx] = val

                prob *= second_prob
                break

        new_factor.add_values([new_values + [prob]])

    return new_factor



def ve(bayes_net, var_query, varlist_evidence): 
    '''
    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by varlist_evidence. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param varlist_evidence: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    '''

    hidden_vars = filter(lambda v: v.name != var_query.name and v not in varlist_evidence, bayes_net.variables())

    new_factors = []
    for factor in bayes_net.factors():
        if len(varlist_evidence) == 0:
            new_factors.append(factor)
            continue

        for var in varlist_evidence:
            if var in factor.scope:
                new_factor = restrict(factor, var, var.dom[var.evidence_index])
                new_factors.append(new_factor)
            else:
                new_factors.append(factor)

    factors = new_factors
    for var in hidden_vars:
        relevant_factors = []
        irrelevant_factors = []
        for factor in factors:
            if var in factor.scope:
                relevant_factors.append(factor)
            else:
                irrelevant_factors.append(factor)

        new_factor = multiply(relevant_factors)
        new_factor = sum_out(new_factor, var)
        factors = irrelevant_factors + [new_factor]

    final_factor = multiply(factors)
    final_factor = normalize(final_factor)
    return final_factor


salary_variable_domains = {
"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
"Gender": ['Male', 'Female'],
"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
"Salary": ['<50K', '>=50K']
}

salary_variable=Variable("Salary", ['<50K', '>=50K'])

def naive_bayes_model(data_file, variable_domains=salary_variable_domains, class_var=salary_variable):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that represents 
    the joint distribution of value assignments to variables in the given dataset.

    Remember a Naive Bayes model assumes P(X1, X2,.... XN, Class) can be represented as 
    P(X1|Class) * P(X2|Class) * .... * P(XN|Class) * P(Class).

    When you generated your Bayes Net, assume that the values in the SALARY column of 
    the dataset are the CLASS that we want to predict.

    Please name the factors as follows. If you don't follow these naming conventions, you will fail our tests.
    - The name of the Salary factor should be called "Salary" without the quotation marks.
    - The name of any other factor should be called "VariableName,Salary" without the quotation marks. 
      For example, the factor for Education should be called "Education,Salary".

    @return a BN that is a Naive Bayes model and which represents the given data set.
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    vars = [Variable(name, domain) for name, domain in variable_domains.items()]
    vars.pop() # remove Salary variable

    factors = []
    for i, v in enumerate(vars):
        factor = Factor(f"{v.name},{class_var.name}", [v, class_var])
        factors.append(factor)

        factor_data = map(lambda row: (row[i], row[-1]), input_data)
        counts = Counter(factor_data)

        total = sum(counts.values())
        values = list(map(lambda i: list(i[0]) + [i[1]/total], counts.items()))
        factor.add_values(values)

    class_var_factor = Factor(class_var.name, [class_var])
    salary_counts = Counter(map(lambda row: row[-1], input_data))
    total = sum(salary_counts.values())
    salary_values = list(map(lambda i: [i[0], i[1]/total], salary_counts.items()))
    class_var_factor.add_values(salary_values)

    vars = [class_var] + vars
    all_factors = [class_var_factor] + factors
    bn = BN("Naive_Bayes_Model_Salary", vars, all_factors)

    return bn


def explore(bayes_net, question):
    '''    
    Return a probability given a Naive Bayes Model and a question number 1-6. 
    
    The questions are below: 
    1. What percentage of the women in the test data set does our model predict having a salary >= $50K? 
    2. What percentage of the men in the test data set does our model predict having a salary >= $50K? 
    3. What percentage of the women in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    4. What percentage of the men in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    5. What percentage of the women in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?
    6. What percentage of the men in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?

    @return a percentage (between 0 and 100)
    ''' 
    ### YOUR CODE HERE ###
    raise NotImplementedError


data_file = "adult-train_tiny.csv"
bn = naive_bayes_model(data_file)
stuff = ve(bn, salary_variable, [])
print(stuff.print_table())