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


def select_reasoning_modules(question, engine):
    reasoning_steps = ["How could I devise an experiment to help solve that problem?",
"Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
"How could I measure progress on this problem?",
"How can I simplify the problem so that it is easier to solve?",
"What are the key assumptions underlying this problem?",
"What are the potential risks and drawbacks of each solution?",
"What are the alternative perspectives or viewpoints on this problem?",
"What are the long-term implications of this problem and its solutions?",
"How can I break down this problem into smaller, more manageable parts?",
"Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating,the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
"Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions,thinking beyond traditional boundaries, and encouraging imagination and originality.",
"Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging thediverse perspectives and expertise of a group to come up with effective solutions.",
"Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
"Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
"Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches."
"What is the core issue or problem that needs to be addressed?",
"What are the underlying causes or factors contributing to the problem?",
"Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
"What are the potential obstacles or challenges that might arise in solving this problem?",
"Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available,and how can they be analyzed?",
"Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
"What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
"How can progress or success in solving the problem be measured or evaluated?",
"What indicators or metrics can be used?",
"Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
"Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
"Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
"Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
"Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
"Is the problem a design challenge that requires creative solutions and innovation?",
"Does the problem require addressing systemic or structural issues rather than just individual instances?",
"Is the problem time-sensitive or urgent, requiring immediate attention and action?",
"What kinds of solution typically are produced for this kind of problem specification?"
"Given the problem specification and the current best solution, have a guess about other possible solutions.",
"Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
"What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
"Ignoring the current best solution, create an entirely new solution to the problem.",
"Let's think step by step.",
"Let's make a step by step plan and implement it with good notion and explanation" ]
    reasoning_steps_formatted = ', '.join([f'"{step}"' for step in reasoning_steps])
    prompt = f"Given the question about SQL generation: '{question}', which of these reasoning modules would be most useful to solve the problem: [{reasoning_steps_formatted}]?"
    selected_modules = connect_gpt(engine, prompt, 100, 0.7, None)  # Assuming a function that handles API calls
    return [module.strip() for module in selected_modules.split(',')]


def adapt_reasoning_modules(selected_modules, question, engine):
    prompt = f"Adapt the following selected reasoning modules for SQL generation: {', '.join(selected_modules)}.\nTask: Generate SQL for the question: '{question}'."
    adapted_modules = connect_gpt(engine, prompt, 200, 0.7, None)
    return adapted_modules

def implement_reasoning_structure(adapted_modules, question, engine):
    prompt = f"Create an actionable reasoning structure for generating SQL based on these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{question}"
    reasoning_structure = connect_gpt(engine, prompt, 300, 0.7, None)
    return reasoning_structure

def implement_sql_generation(question, reasoning_structure, schema_prompt, engine):
    prompt = f"Using the actionable reasoning structure: {reasoning_structure}\nGenerate an SQL query for the task:{'question'}.\nSchema Details:\n{schema_prompt}"
    sql_query = connect_gpt(engine, prompt, 300, 0.7, [';'])
    return sql_query.strip()


def connect_gpt(prompt, engine, max_tokens, temperature=0.5, stop=None):
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        return response['choices'][0]['text'].strip()  # Correctly accessing the 'text' key
    except Exception as e:
        return f"Error: {str(e)}"



def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None):
    openai.api_key = api_key
    response_list = []
    reasoning_output_list = []

    with open('reasoning_outputs.txt', 'w') as reasoning_file:
        for i, question in enumerate(question_list):
            print(f'--------------------- processing {i}th question ---------------------')
            print(f'The question is: {question}')

            db_path = db_path_list[i]
            knowledge = knowledge_list[i] if knowledge_list else None
            
            # Process reasoning steps
            selected_modules = select_reasoning_modules(question, engine)
            adapted_modules = adapt_reasoning_modules(selected_modules, question, engine)
            reasoning_structure = implement_reasoning_structure(adapted_modules, question)
            schema_prompt = generate_schema_prompt(db_path)
            sql_query = implement_sql_generation(question, reasoning_structure, schema_prompt, engine)
            
            # Writing reasoning steps output to file
            reasoning_output = {
                'question': question,
                'selected_modules': selected_modules,
                'adapted_modules': adapted_modules,
                'reasoning_structure': reasoning_structure
            }
            reasoning_file.write(json.dumps(reasoning_output) + '\n')
            
            # Prepare SQL output
            db_id = db_path.split('/')[-1].split('.sqlite')[0]
            formatted_sql = sql_query + '\t----- bird -----\t' + db_id
            response_list.append({'db_id': db_id, 'sql_query': formatted_sql})
    return response_list

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
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = db_root_path + data['db_id'] + '/' + data['db_id'] +'.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
    
    return question_list, db_path_list, knowledge_list

def generate_sql_file(sql_lst, output_path=None):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql
    
    if output_path:
        directory_path = os.path.dirname(output_path)  
        new_directory(directory_path)
        json.dump(result, open(output_path, 'w'), indent=4)
    
    return result    

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--mode', type=str, default='dev')
    args_parser.add_argument('--test_path', type=str, default='')
    args_parser.add_argument('--use_knowledge', type=str, default='False')
    args_parser.add_argument('--db_root_path', type=str, default='')
    # args_parser.add_argument('--db_name', type=str, required=True)
    args_parser.add_argument('--api_key', type=str, required=True)
    args_parser.add_argument('--engine', type=str, required=True, default='code-davinci-002')
    args_parser.add_argument('--data_output_path', type=str)
    args_parser.add_argument('--chain_of_thought', type=str)
    args = args_parser.parse_args()
    
    eval_data = json.load(open(args.eval_path, 'r'))
    # '''for debug'''
    # eval_data = eval_data[:3]
    # '''for debug'''
    
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    assert len(question_list) == len(db_path_list) == len(knowledge_list)
    
    if args.use_knowledge == 'True':
        responses = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list, api_key=args.api_key, engine=args.engine, knowledge_list=knowledge_list)
    else:
        responses = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list, api_key=args.api_key, engine=args.engine, knowledge_list=None)
    
    if args.chain_of_thought == 'True':
        output_name = args.data_output_path + 'predict_' + args.mode + '_cot.json'
    else:
        output_name = args.data_output_path + 'predict_' + args.mode + '.json'
    # pdb.set_trace()
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print('successfully collect results from {} for {} evaluation; Use knowledge: {}; Use COT: {}'.format(args.engine, args.mode, args.use_knowledge, args.chain_of_thought))
