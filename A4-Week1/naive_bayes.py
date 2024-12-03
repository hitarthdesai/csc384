############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 4 Starter Code
## v1.0
############################################################

from collections import Counter
from bnetbase import Variable, Factor, BN
import csv
from itertools import product

def convert_factor_table_key_to_tuple(factor):
    def get_table(factor):
        saved_values = []  #save and then restore the variable assigned values.

        for v in factor.scope:
            saved_values.append(v.get_assignment_index())

        prob_dict = {}
        get_values_recursive(factor, factor.scope, prob_dict)

        for v in factor.scope:
            v.set_assignment_index(saved_values[0])
            saved_values = saved_values[1:]

        return prob_dict
    
    def get_values_recursive(factor, vars, info_dict):
        if len(vars) == 0:
            newkey = ""
            for v in factor.scope:
                newkey += "{} = {},".format(v.name, v.get_assignment())
            info_dict[newkey] = factor.get_value_at_current_assignments()
        else:
            for val in vars[0].domain():
                vars[0].set_assignment(val)
                get_values_recursive(factor, vars[1:], info_dict)
    
    factor_table = get_table(factor)
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

    factor_table = convert_factor_table_key_to_tuple(factor)
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
    factor_table = convert_factor_table_key_to_tuple(factor)
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
    factor_table = convert_factor_table_key_to_tuple(factor)
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

    num_entries_cartesian_product = 1
    for v in new_factor_scope:
        num_entries_cartesian_product *= len(v.domain())

    cartesian_product = product(*[v.domain() for v in new_factor_scope])
    new_values = []
    for entry in cartesian_product:
        new_values.append(list(entry) + [1])


    for factor in factor_list:
        factor_table = convert_factor_table_key_to_tuple(factor)

        indices = []
        for f in factor.scope:
            for i, v in enumerate(new_factor_scope):
                if v.name == f.name:
                    indices.append(i)
                
        for key, prob in factor_table.items():

            for entry in new_values:
                match = True
                for i, v in enumerate(key):
                    if entry[indices[i]] != v:
                        match = False
                        break

                if match:
                    entry[-1] *= prob
            
    new_factor.add_values(new_values)
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

    new_factors = []
    for factor in bayes_net.factors():
        for var in varlist_evidence:
            if var not in factor.scope:
                continue
            factor = restrict(factor, var, var.get_evidence())
        new_factors.append(factor)

    factors_to_consider = new_factors
    hidden_vars = filter(lambda v: v != var_query and v not in varlist_evidence, bayes_net.variables())
    for var in hidden_vars:
        relevant_factors = list(filter(lambda f: var in f.scope, factors_to_consider))
        if len(relevant_factors) == 0:
            continue

        factors_to_consider = list(filter(lambda f: f not in relevant_factors, factors_to_consider))
        multiplied_factor = multiply(relevant_factors)
        summed_out_factor = sum_out(multiplied_factor, var)
        factors_to_consider.append(summed_out_factor)

    finally_multiplied_factor = multiply(factors_to_consider)
    final_factor = normalize(finally_multiplied_factor)
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

    class_var_factor = Factor(class_var.name, [class_var])
    salary_counts = Counter(map(lambda row: row[-1], input_data))
    total = sum(salary_counts.values())
    salary_values = list(map(lambda i: [i[0], i[1]/total], salary_counts.items()))
    class_var_factor.add_values(salary_values)

    factors = []
    for i, v in enumerate(vars):
        factor = Factor(f"{v.name},{class_var.name}", [v, class_var])
        factors.append(factor)

        factor_data = map(lambda row: (row[i], row[-1]), input_data)
        counts = Counter(factor_data)
        for s in variable_domains["Salary"]:
            for value in v.domain():
                if (value, s) not in counts:
                    counts[(value, s)] = 0

        for c in counts:
            counts[c] = counts[c] / salary_counts[c[1]]
        
        values = list(map(lambda i: list(i[0]) + [i[1]], counts.items()))
        factor.add_values(values)

    vars = [class_var] + vars
    all_factors = [class_var_factor] + factors
    bn = BN("Naive_Bayes_Model_Salary", vars, all_factors)

    return bn

def explore_question_1(bayes_net):
    """
    What percentage of the women in the test data set does our model predict having a salary >= $50K?
    """
    salary_var = bayes_net.get_variable("Salary")
    evidence_vars = [
        bayes_net.get_variable("Work"),
        bayes_net.get_variable("Education"),
        bayes_net.get_variable("Occupation"),
        bayes_net.get_variable("Relationship"),
    ]

    female_count, female_high_salary_count = 0, 0
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        gender_idx = headers.index("Gender")

        for row in reader:
            if row[gender_idx] == "Male":
                continue

            female_count += 1
            for i, val in enumerate(row):
                if headers[i] == "Salary" or headers[i] in "Gender":
                    continue
                bayes_net.get_variable(headers[i]).set_evidence(val)

            salary_dist = ve(bayes_net, salary_var, evidence_vars)
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k = salary_dist.get_value_at_current_assignments()
            if prob_salary_ge_50k > 0.5:
                female_high_salary_count += 1

    return (female_high_salary_count / female_count) * 100

def explore_question_2(bayes_net):
    """
    What percentage of the men in the test data set does our model predict having a salary >= $50K?
    """
    salary_var = bayes_net.get_variable("Salary")
    evidence_vars = [
        bayes_net.get_variable("Work"),
        bayes_net.get_variable("Education"),
        bayes_net.get_variable("Occupation"),
        bayes_net.get_variable("Relationship"),
    ]

    male_count, male_high_salary_count = 0, 0
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        gender_idx = headers.index("Gender")

        for row in reader:
            if row[gender_idx] == "Female":
                continue

            male_count += 1
            for i, val in enumerate(row):
                if headers[i] == "Salary" or headers[i] == "Gender":
                    continue
                bayes_net.get_variable(headers[i]).set_evidence(val)

            salary_dist = ve(bayes_net, salary_var, evidence_vars)
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k = salary_dist.get_value_at_current_assignments()
            if prob_salary_ge_50k > 0.5:
                male_high_salary_count += 1

    return (male_high_salary_count / male_count) * 100

def explore_question_3(bayes_net):
    """
    What percentage of the women in the test data set satisfies the condition: P(Salary=">=$50K"| Evidence) > P(Salary=">=$50K" | Evidence, Gender)
    """
    salary_var = bayes_net.get_variable("Salary")
    gender_var = bayes_net.get_variable("Gender")
    evidence_vars = [
        bayes_net.get_variable("Work"),
        bayes_net.get_variable("Education"),
        bayes_net.get_variable("Occupation"),
        bayes_net.get_variable("Relationship"),
    ]

    female_count, sat_count = 0, 0
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        gender_idx = headers.index("Gender")

        for row in reader:
            if row[gender_idx] == "Male":
                continue

            female_count += 1
            for i, val in enumerate(row):
                if headers[i] == "Salary" or headers[i] in "Gender":
                    continue
                bayes_net.get_variable(headers[i]).set_evidence(val)

            salary_dist = ve(bayes_net, salary_var, evidence_vars)
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k = salary_dist.get_value_at_current_assignments()

            gender_var.set_evidence("Female")
            salary_dist = ve(bayes_net, salary_var, evidence_vars + [gender_var])
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k_given_female = salary_dist.get_value_at_current_assignments()
            
            if prob_salary_ge_50k > prob_salary_ge_50k_given_female:
                sat_count += 1

    return (sat_count / female_count) * 100

def explore_question_4(bayes_net):
    """
    What percentage of the men in the test data set satisfies the condition: P(S=">=50K"|Evidence) is strictly greater than P(S=">=50K"|Evidence,Gender)?
    """
    salary_var = bayes_net.get_variable("Salary")
    gender_var = bayes_net.get_variable("Gender")
    evidence_vars = [
        bayes_net.get_variable("Work"),
        bayes_net.get_variable("Education"),
        bayes_net.get_variable("Occupation"),
        bayes_net.get_variable("Relationship"),
    ]

    male_count, sat_count = 0, 0
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        gender_idx = headers.index("Gender")

        for row in reader:
            if row[gender_idx] == "Female":
                continue

            male_count += 1
            for i, val in enumerate(row):
                if headers[i] == "Salary" or headers[i] in "Gender":
                    continue
                bayes_net.get_variable(headers[i]).set_evidence(val)

            salary_dist = ve(bayes_net, salary_var, evidence_vars)
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k = salary_dist.get_value_at_current_assignments()

            gender_var.set_evidence("Male")
            salary_dist = ve(bayes_net, salary_var, evidence_vars + [gender_var])
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k_given_male = salary_dist.get_value_at_current_assignments()
            
            if prob_salary_ge_50k > prob_salary_ge_50k_given_male:
                sat_count += 1

    return (sat_count / male_count) * 100

def explore_question_5(bayes_net):
    """
    What percentage of the women in the test data set with a predicted salary over $50K (P(Salary=">=50K"|E) > 0.5) have an actual salary over $50K?
    """
    salary_var = bayes_net.get_variable("Salary")
    evidence_vars = [
        bayes_net.get_variable("Work"),
        bayes_net.get_variable("Education"),
        bayes_net.get_variable("Occupation"),
        bayes_net.get_variable("Relationship"),
    ]

    female_high_salary_count, sat_count = 0, 0
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        gender_idx = headers.index("Gender")

        for row in reader:
            if row[gender_idx] == "Male":
                continue

            is_salary_ge_50k = False
            for i, val in enumerate(row):
                if headers[i] == "Gender":
                    continue
                if headers[i] == "Salary":
                    is_salary_ge_50k = val == ">=50K"
                    continue
                bayes_net.get_variable(headers[i]).set_evidence(val)

            salary_dist = ve(bayes_net, salary_var, evidence_vars)
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k = salary_dist.get_value_at_current_assignments()

            if prob_salary_ge_50k > 0.5:
                female_high_salary_count += 1
                if is_salary_ge_50k:
                    sat_count += 1

    return (sat_count / female_high_salary_count) * 100

def explore_question_6(bayes_net):
    """
    What percentage of the men in the test data set with a predicted salary over $50K (P(Salary=">=50K"|E) > 0.5) have an actual salary over $50K?
    """
    salary_var = bayes_net.get_variable("Salary")
    evidence_vars = [
        bayes_net.get_variable("Work"),
        bayes_net.get_variable("Education"),
        bayes_net.get_variable("Occupation"),
        bayes_net.get_variable("Relationship"),
    ]

    male_high_salary_count, sat_count = 0, 0
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        gender_idx = headers.index("Gender")

        for row in reader:
            if row[gender_idx] == "Female":
                continue

            is_salary_ge_50k = False
            for i, val in enumerate(row):
                if headers[i] == "Gender":
                    continue
                if headers[i] == "Salary":
                    is_salary_ge_50k = val == ">=50K"
                    continue
                bayes_net.get_variable(headers[i]).set_evidence(val)

            salary_dist = ve(bayes_net, salary_var, evidence_vars)
            salary_var.set_assignment(">=50K")
            prob_salary_ge_50k = salary_dist.get_value_at_current_assignments()

            if prob_salary_ge_50k > 0.5:
                male_high_salary_count += 1
                if is_salary_ge_50k:
                    sat_count += 1

    return (sat_count / male_high_salary_count) * 100

explore_question_functions = {
    1: explore_question_1,
    2: explore_question_2,
    3: explore_question_3,
    4: explore_question_4,
    5: explore_question_5,
    6: explore_question_6
}

def explore(bayes_net, question):
    return explore_question_functions[question](bayes_net)

if __name__ == '__main__':
    data_file = "data/adult-train.csv"
    bn = naive_bayes_model(data_file)
