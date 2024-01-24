#!/usr/bin/env python3
import argparse
import fnmatch
import json
import os
import pdb
import pickle
import re
import sqlite3
from typing import Dict, List, Tuple

import backoff
import openai
import pandas as pd
import sqlparse
from tqdm import tqdm
'''openai configure'''

openai.debug=True


def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  


def get_db_schemas(bench_root: str, db_name: str) -> Dict[str, str]:
    """
    Read an sqlite file, and return the CREATE commands for each of the tables in the database.
    """
    asdf = 'database' if bench_root == 'spider' else 'databases'
    with sqlite3.connect(f'file:{bench_root}/{asdf}/{db_name}/{db_name}.sqlite?mode=ro', uri=True) as conn:
        # conn.text_factory = bytes
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            schemas[table[0]] = cursor.fetchone()[0]

        return schemas

def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print(header)
    # Print the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    '''
    :param root_place:
    :param db_name:
    :return:
    '''
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."
    # question_prompt = "-- {}".format(question) + '\n SELECT '
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge_prompt:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt

def cot_wizard():
    cot = "\nGenerate the SQL after thinking step by step: "
    
    return cot

def few_shot():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    birth_year  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge: age = year - birth_year;\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. referring to external knowledge, we need to filter singers 'by year' - 'birth_year' > 27; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE year - birth_year > 27;</s>"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + ini_cot_result
    
    return one_shot_demo

def few_shot_no_kg():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    age  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge:\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. 'older than 27' refers to age > 27 in SQL; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE age > 27;</s>"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + ini_cot_result
    
    return one_shot_demo

def generate_simple_prompt(db_path, question, knowledge=None):
    return generate_combined_prompts_one(db_path, question, knowledge)

# New function to generate prompt for moderate difficulty
def generate_moderate_prompt(db_path, question, knowledge=None):
    basic_prompt = generate_combined_prompts_one(db_path, question, knowledge)
    moderate_prompt = basic_prompt + "\n-- For moderate difficulty, consider table relationships and necessary joins."
    # Add more specific guidance based on your observations
    return moderate_prompt

# New function to generate prompt for difficult difficulty
def generate_difficult_prompt(db_path, question, knowledge=None):
    basic_prompt = generate_combined_prompts_one(db_path, question, knowledge)
    difficult_prompt = basic_prompt + "\n-- For difficult questions, consider complex conditions, subqueries, and advanced SQL features."
    # Add more detailed hints or steps based on your requirements
    return difficult_prompt




def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=None) # This is the entry to collect values
    comment_prompt = generate_comment_prompt(question, knowledge)

    combined_prompts = schema_prompt + '\n\n' + comment_prompt + cot_wizard() + '\nSELECT '
    # combined_prompts = few_shot() + '\n\n' + schema_prompt + '\n\n' + comment_prompt

    # print(combined_prompts)

    return combined_prompts

def quota_giveup(e):
    return isinstance(e, openai.error.RateLimitError) and "quota" in str(e)

@backoff.on_exception(
    backoff.constant,
    openai.error.OpenAIError,
    giveup=quota_giveup,
    raise_on_giveup=True,
    interval=20
)

import json
import os

def extract_correct_sql_from_json(eval_path: str) -> list:
    # Check if the file exists
    if not os.path.exists(eval_path):
        raise Exception(f"File not found: {eval_path}")

    try:
        with open(eval_path, 'r') as file:
            data_json = json.load(file)
        # Print the loaded JSON data for debugging
        print("Loaded JSON data:", data_json)
    except IOError as e:
        raise IOError(f"Could not open file: {eval_path}. Error: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"File is not valid JSON: {eval_path}. Error: {e}")

    correct_sql_list = []
    for data in data_json:
        # Check for 'SQL' key in each entry
        if 'SQL' in data:
            print(f"SQL key found for question ID {data.get('question_id')}")
            correct_sql = data['SQL']
        else:
            print(f"SQL key missing for question ID {data.get('question_id')}")
            correct_sql = None

        correct_sql_list.append(correct_sql)

    # Print the entire list of extracted SQL queries for further verification
    print("All extracted SQL queries:", correct_sql_list)

    return correct_sql_list


import sqlparse

def normalize_sql(sql):
    """ Normalize SQL query for comparison. """
    sql = sql.lower()
    sql = sqlparse.format(sql, reindent=True, keyword_case='upper')
    return ' '.join(sql.split())

import difflib


import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

def get_sql_components(sql):
    """ Breaks down an SQL query into its constituent components. """
    parsed = sqlparse.parse(sql)[0]
    components = {}
    for token in parsed.tokens:
        if token.ttype is DML:
            components['DML'] = token.value
        if token.ttype is Keyword:
            components[token.value.upper()] = []
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                components[token.get_real_name()].append(identifier.value)
        elif isinstance(token, Identifier):
            components[token.get_real_name()] = [token.value]
    return components

import sqlparse
from sqlparse.tokens import Keyword, Whitespace

def enhanced_get_sql_components(sql):
    """
    Enhanced function to break down an SQL query into its constituent components,
    handling a wider range of SQL structures and syntaxes.
    """
    parsed = sqlparse.parse(sql)[0]
    components = {
        'SELECT': [],
        'FROM': [],
        'WHERE': [],
        'JOIN': [],
        'GROUP_BY': [],
        'ORDER_BY': [],
        'HAVING': [],
        'LIMIT': [],
        'OTHERS': []
    }

    current_component = 'OTHERS'
    for token in parsed.tokens:
        if token.is_group:
            for subtoken in token.tokens:
                if subtoken.ttype is Keyword:
                    current_component = subtoken.normalized.upper()
                    if current_component not in components:
                        current_component = 'OTHERS'
                elif subtoken.ttype is not Whitespace:
                    components[current_component].append(subtoken.normalized)
        elif token.ttype is Keyword:
            current_component = token.normalized.upper()
            if current_component not in components:
                current_component = 'OTHERS'
        elif token.ttype is not Whitespace:
            components[current_component].append(token.normalized)

    # Normalize components to handle variations that do not affect functionality
    for key in components:
        components[key] = sorted(set(components[key]))

    return components

def refined_calculate_sql_similarity(sql1, sql2):
    """
    Refined function to calculate similarity based on SQL components,
    accounting for functional equivalences rather than exact matches.
    """
    components1 = enhanced_get_sql_components(sql1)
    components2 = enhanced_get_sql_components(sql2)

    total_components = set(components1.keys()) | set(components2.keys())
    similarity_count = 0

    for component in total_components:
        if sorted(components1.get(component, [])) == sorted(components2.get(component, [])):
            similarity_count += 1

    return (similarity_count / len(total_components)) * 100 if total_components else 0




def calculate_similarity_percentage(str1, str2):
    """ Calculate the similarity percentage between two strings. """
    return difflib.SequenceMatcher(None, str1, str2).ratio() * 100

def semantic_comparison(generated_sql, correct_sql, db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            # Execute and fetch results from the generated SQL query
            gen_cursor = conn.cursor()
            gen_cursor.execute(generated_sql)
            gen_results = gen_cursor.fetchall()

            # Execute and fetch results from the correct SQL query
            corr_cursor = conn.cursor()
            corr_cursor.execute(correct_sql)
            corr_results = corr_cursor.fetchall()

            return gen_results == corr_results
    except Exception as e:
        print(f"Error during semantic comparison: {e}")
        return False
    



def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    # print(prompt)
    try:
        result = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
    except Exception as e:
        result = 'error:{}'.format(e)
    return result



def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None, eval_path=None):
    responses_dict = {}
    response_list = []
    openai.api_key = api_key

    # Use the function to get the correct SQL queries
    hardcoded_json_path = '/home/shouvon/DAMO-ConvAI/bird/llm/data/dev/dev2.json' 
    correct_sql_list = extract_correct_sql_from_json(hardcoded_json_path) if hardcoded_json_path else [None] * len(question_list)

    feedback_results = {}
    for i, question in tqdm(enumerate(question_list)):
        difficulty = difficulty_list[i]
        print(f"Processing {i}th question (Difficulty: {difficulty}): {question}")
        if knowledge_list:
            print(f"External Knowledge: {knowledge_list[i]}")

        # Generate the prompt based on difficulty
        if difficulty == 'simple':
            cur_prompt = generate_simple_prompt(db_path=db_path_list[i], question=question, knowledge=knowledge_list[i] if knowledge_list else None)
        elif difficulty == 'moderate':
            cur_prompt = generate_moderate_prompt(db_path=db_path_list[i], question=question, knowledge=knowledge_list[i] if knowledge_list else None)
        elif difficulty == 'difficult':
            cur_prompt = generate_difficult_prompt(db_path=db_path_list[i], question=question, knowledge=knowledge_list[i] if knowledge_list else None)
        else:
            cur_prompt = generate_simple_prompt(db_path=db_path_list[i], question=question, knowledge=knowledge_list[i] if knowledge_list else None)


        attempt = 0
        is_normally_correct = False
        while attempt < 3 and not is_normally_correct:
            print(f"Processing {i}th question, attempt {attempt+1}: {question}")
            if knowledge_list:
                print(f"External Knowledge: {knowledge_list[i]}")
            plain_result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0, stop=['--', '\n\n', ';', '#'])

            sql = plain_result if type(plain_result) == str else 'SELECT' + plain_result['choices'][0]['text']
            db_id = db_path_list[i].split('/')[-1].split('.sqlite')[0]
            sql = sql + '\t----- bird -----\t' + db_id

            print(f"Extracted SQL: {sql}")

            if correct_sql_list[i]:
                print(f"Correct SQL (Ground Truth): {correct_sql_list[i]}")
            else:
                print("Correct SQL (Ground Truth) not available")

            similarity_percentage = 0
            if correct_sql_list[i]:
                similarity_percentage = calculate_similarity_percentage(normalize_sql(sql), normalize_sql(correct_sql_list[i]))
                is_normally_correct = similarity_percentage >= 50

            attempt += 1

        is_semantically_correct = False
        if is_normally_correct:
            is_semantically_correct = semantic_comparison(sql, correct_sql_list[i], db_path_list[i])

        feedback_results[question] = {"generated_sql": sql, 
                                      "is_normally_correct": is_normally_correct, 
                                      "similarity_percentage": similarity_percentage,
                                      "is_semantically_correct": is_semantically_correct, 
                                      "attempts": attempt}

        response_list.append(sql)

    return response_list, feedback_results






def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data['question'])

    return question_list

def knowledge_package(data_json, knowledge=False):
    knowledge_list = []
    for data in data_json:
        knowledge_list.append(data['evidence'])

    return knowledge_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    difficulty_list = [] 
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = db_root_path + data['db_id'] + '/' + data['db_id'] +'.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
        difficulty_list.append(data['difficulty'])
    
    return question_list, db_path_list, knowledge_list, difficulty_list


def generate_sql_file(sql_lst, output_path=None):
    sql_dict = {}
    for i, sql in enumerate(sql_lst):
        sql_query = sql.split("\t----- bird -----")[0].strip()  # Extract only the SQL part
        sql_dict[i] = sql_query
    
    if output_path:
        directory_path = os.path.dirname(output_path)  
        new_directory(directory_path)
        with open(output_path, 'w') as f:
            json.dump(sql_dict, f, indent=4)
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql
    
    if output_path:
        directory_path = os.path.dirname(output_path)  
        new_directory(directory_path)
        json.dump(result, open(output_path, 'w'), indent=4)
    
    return result    


def save_feedback(feedback_results, output_path):
    with open(output_path, 'w') as file:
        for question, result in feedback_results.items():
            feedback_str = f"Generated SQL for '{question}' is "
            feedback_str += "correct both normally and semantically." if result['is_normally_correct'] and result['is_semantically_correct'] else "incorrect."
            feedback_str += f" Similarity: {result['similarity_percentage']}%."
            if result['attempts'] == 3 and not result['is_normally_correct']:
                feedback_str += " Maximum attempts reached."
            file.write(feedback_str + '\n')



if __name__ == '__main__':
    # Argument parsing
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--mode', type=str, default='dev')
    args_parser.add_argument('--test_path', type=str, default='')
    args_parser.add_argument('--use_knowledge', type=str, default='False')
    args_parser.add_argument('--db_root_path', type=str, default='')
    args_parser.add_argument('--api_key', type=str, required=True)
    args_parser.add_argument('--engine', type=str, required=True, default='code-davinci-002')
    args_parser.add_argument('--data_output_path', type=str)
    args_parser.add_argument('--chain_of_thought', type=str)
    args = args_parser.parse_args()
    
    # Load evaluation data
    eval_data = json.load(open(args.eval_path, 'r'))
    
    # Decouple question and schema
    question_list, db_path_list, knowledge_list, difficulty_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    # Collect responses from GPT
    if args.use_knowledge == 'True':
        responses, feedback_results = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list, difficulty_list=difficulty_list, api_key=args.api_key, engine=args.engine, knowledge_list=knowledge_list)
    else:
        responses, feedback_results = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list,  difficulty_list=difficulty_list, api_key=args.api_key, engine=args.engine, knowledge_list=None)
    
    # Save SQL queries
    output_name = args.data_output_path + 'predict_' + args.mode + ('_cot.json' if args.chain_of_thought == 'True' else '.json')
    generate_sql_file(sql_lst=responses, output_path=output_name)

    # Save feedback
    feedback_file_path = './feedback_results.txt'
    save_feedback(feedback_results, feedback_file_path)

    print('Successfully collected results from {} for {} evaluation; Use knowledge: {}; Use COT: {}'.format(args.engine, args.mode, args.use_knowledge, args.chain_of_thought))