import difflib
import json
import os
import re
import sys
import time
import traceback
from io import BytesIO
import datetime
import boto3
import openai
import pandas as pd
import pdfplumber
import timeout_decorator
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_ORG = os.getenv("OPENAI_API_ORG")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.organization = OPENAI_API_ORG
openai.api_key = OPENAI_API_KEY  # os.getenv("OPENAI_API_KEY")

# Set the option to display all columns. None means unlimited.
pd.set_option('display.max_columns', None)
# Set the option to display all rows. None means unlimited.
pd.set_option('display.max_rows', None)
# Increase the width of each column to prevent truncation
pd.set_option('display.max_colwidth', None)

# new code part to scan pdf
final_output = {}
doc_parts = 30  # in how many parts should the doc be read
gpt_model = 'gpt-3.5-turbo'
# Set the path to your local PDF file
# pdfpath = "oldbpr.pdf"  # Update this with your local file path
def writetos3(aws_access_key_id, aws_secret_access_key, data_to_write,s3_file_name):
    region = 'ap-south-1'
    bucket = 'snuckworks'
    try:
        data_to_write = json.dumps(data_to_write)
        s3_client = boto3.client('s3',region_name=region ,aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
        s3_client.put_object(Bucket=bucket, Key=s3_file_name, Body=data_to_write)
        return "wrote final output to s3.\n"
    except:
        return f"error writing to s3: {traceback.format_exc()}"


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

def find_base_number(number):
    """
    This function is to make doc_part dynamic.
    :param number: The total pages of the BPR
    :return: updated DOc_part
    """
    if number % 10 < 5:
        base_number = (number // 10) * 10  # Lower base number
    else:
        base_number = ((number // 10) + 1) * 10  # Higher base number
    return base_number

def extract_sections_with_page_numbers(text, value_to_search):
    """
    This function reads the index page of the bpr model and extracts the sections with their page numbers
    :param text: Input text of the index page
    :return: For BPR it returns the page number of starting and ending of auditor's section
    """
    sections = {}

    lower_text = text.lower()
    toc_text = lower_text.split("table of content")[-1]
    toc_lines = toc_text.split('\n')
    for line in toc_lines:
        text_without_dots = line.replace(".", "")
        # print("LINE", text_without_dots)
        try:
            parts = text_without_dots.split()
            # print("Section", parts)
            try:
                value = int(parts[-1])
                company_name = ' '.join(parts[:-1])
                sections[company_name] = int(value)
            except:
                # print("cant convert to int")
                company_name = ' '.join(parts[:-2])
                value = parts[-2]
                sections[company_name] = int(value)
        except:
            pass
    # print(sections)
    try:
        for index, (key, value) in enumerate(sections.items()):
            if value_to_search in key.lower():
                value1 = list(sections.values())[index]
                value2 = list(sections.values())[index+1]
                break
        return value1, value2
        # index = list(sections.keys()).index('AUDITOR’S OBSERVATIONS')
        # value1 = list(sections.values())[index]
        # value2 = list(sections.values())[index+1]
    except:
        return 0, 0
    # print(index)

log_str=''


def getOldReport(pdf_path, comp_id='', report_id='', report_type = "GPR"):
    global log_str
    global final_output
    log_path = f"../../temp/logs/pythonOldReport_{report_id}.log"

    s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    response = s3.get_object(Bucket="snuckworks", Key=pdf_path)

    # pdf_content = response['Body'].read()
    pdf_content = response['Body'].read()

    with BytesIO(pdf_content) as pdf_file:
        with pdfplumber.open(pdf_file) as pdf:
            pages=[]
            page_text = ""
            pg_num_content = 2
            for page_number, page in enumerate(pdf.pages):
                pp=""
                time.sleep(1)
                # Extract text from the current page
                text = page.extract_text()
                lower_text = text.lower()
                # print(f"Content from page {page_number}:")
                # print(text)
                # print("-----")
                if ('Glossary of Key Ratios and Computation' in text):
                    ## stop adding to text.. now the report is at its end and showing generic text
                    # print(f"Breaking after page number {page_number}")
                    # print(text)
                    break

                if len(text) > 0:
                    # only append if there is data
                    page_text += f"#T#\n{text}\n#TE#\n"
                    pp += f"#T#\n{text}\n#TE#\n"

                    if "table of content" in lower_text:
                        pg_num_content = page_number

                if report_type == "GPR":
                    tables = page.extract_tables()
                    for table_number, table in enumerate(tables):
                        # Convert table to a DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0])
                        if (df.shape[0] > 0) & (df.shape[1] > 0):
                            if (df.iloc[0,0] != ''):
                                ## only append if table has any data
                                df.columns = [col if col is not None else 'NoLabel' for col in df.columns]
                                #print(df)
                                table_data='@'.join(df.columns) + "***"
                                for index, row in df.iterrows():
                                    table_data += '@'.join(str(value) for value in row) + "***"

                                page_text += f"-D-\n{table_data}\n-DE-\n"

                pages.append(pp)
    # Do something with page_text, like printing or saving to a log file
    # print(page_text)
    # print(len(pages))
    doc_parts = find_base_number(len(pages))
    # print("doc_parts: ", doc_parts)
    # print(len(page_text))
    def get_component_wise_data(i, j, pages):
        page_text_data = ""
        for x in range(i, j):
            page_text_data += pages[x]

        return page_text_data

    timeout = 30
    def update_pagetext(pages, company_name):
        """
        This function removes company name from the end of the page, and update the page text
        :param pages:
        :param company_name:
        :return: updated page_text
        """
        # print(company_name)
        # print(len(company_name))
        for i in range(1, len(pages)):
            # print("Length: ", len(company_name)+30)
            pages[i] = pages[i][:-(len(company_name)+30)]

        page_text=""
        for i in range(len(pages)):
            page_text += pages[i]

        return page_text
    
    def get_audit_result(i, j, pages):
        output = {}
        page_text_auditor = ""
        for x in range(i, j):
            page_text_auditor += pages[x]
        parts = page_text_auditor.split(
            "Observation as per auditor’s report for the year ended")

        for i, part in enumerate(parts[1:], start=1):
            year = part.split()[2]
            if ":" in year:
                year = int(part.split()[2][:-1])  # Extracting the year from the text
            else:
                year = int(part.split()[2])
            current_year = datetime.datetime.now().year
            if year >= current_year-2:
                output[year] = parts[i]
            else:
                continue

        if len(output) == 0:
            """
            Auditor's section dosenot exist in the BPR
            """
            return output

        return output
    
    def get_needed_format(input_data, auditors_system_prompt):
        global gpt_model
        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{auditors_system_prompt}"},
                {"role": "user", "content": f"{input_data}"}
            ]
        )

        return gpt_response.choices[0].message['content']

    def answer_checker(system_prompt, response_list, max_tokens):
        global gpt_model
        global timeout

        checker_prompt = """
                * Do not use previous insturctions/references/answers to complete this task
                * given is a prompt given to chatgpt for generating some financial information about a company. it starts with "--PROMPT"
                * this prompt was run a couple of times on input data. 
                * given after the prompt is a list containing few responses received from chatgpt through this prompt. it starts with "--RESPONSE"
                * Your task is to understand the prompt, then look at all responses given as a Python list, and select the index of the most suitable response from the list.
                * IMPORTANT: INDEX OF THE INPUT PYTHON LIST STARTS AT 0.
                * Use following guidelines to choose the most suitabile response:
                a) iterate over the list, pick one response at a time
                b) a response will be a json, and can have child jsons, and lists
                c) a suitable response will have least number of list or values blanks or like: '', or None, or NA
                d) it will seem to be have the right answers according to the input prompt
                e) if more than one responses seem suitable, choose any one
                f) Then pick its index and return answer in this format: [<index>]
            """
        checker_prompt += "\n" + "--PROMPT\n" + system_prompt + "\n"
        for idx, j in enumerate(response_list):
            checker_prompt += f"--RESPONSE {idx + 1}\n"
            checker_prompt += str(response_list[idx]) + "\n"

        user_content = ''
        # print(f"response list ------ {response_list}")
        # response -> answer -> final
        ans_idx = 1
        pattern = r'\d+'
        while ans_idx <= 2:
            try:
                timeout = 60
                final_response = run_llm(
                    user_content, checker_prompt, gpt_model, max_tokens)
                # print(f"answer check ---- {final_response['choices'][0]['message']['content']}")
                if 'timeouterror' not in final_response.keys():
                    response = final_response['choices'][0]['message']['content']
                    # print(f"index answer {response}")
                    # find the index
                    matches = re.findall(pattern, str(response))
                    matches_found = len(matches)
                    if matches_found == 0:
                        # print("No index given. Retrying")
                        raise ValueError("No index given. Retrying")
                    if matches_found > 1:
                        # print("More than one index given. Retrying")
                        raise ValueError("More than one index given. Retrying")
                    if matches_found == 1:
                        # sometimes it starts index at 1 even though it was asked to start at 0
                        if int(matches[0]) == len(response_list):
                            response = int(matches[0]) - 1
                        else:
                            response = int(matches[0])

                    # print(f"rsponse {response}")

                    first_level_response = response_list[response]
                    # print(f"final answer {first_level_response}")
                    update_output(first_level_response)
                    # print("#########################")
                else:
                    pass
                break
            except Exception as e:
                # print(f"Error setting correct answer: {e}")
                traceback.print_exc()
                # print(system_prompt)
                # print(response_list)
            ans_idx += 1

    def update_output(inp):
        global final_output
        global log_str
        # print(f"----- receivd final response {inp}")
        log_str += f"----- receivd final response {inp}\\n"
        # print(f"----- receivd final response {inp}\\n")
        if isinstance(inp, list):
            for k in inp.keys():
                if len(inp[k]) > 0:
                    if inp[k][0] != '':
                        final_output[k] = inp[k]
        elif isinstance(inp, dict):
            for k in inp.keys():
                # print(f"inp k {inp[k]}")
                if (inp[k] == '') | (inp[k] == ['']):
                    pass
                if isinstance(inp[k], dict):
                    for childk in inp[k].keys():
                        if k in final_output.keys():
                            # print(f"k is {k}")
                            # print(f"childk is {childk}")
                            # print(type(final_output[k]), type(k), type(childk))
                            # print(f"final output is {final_output[k]}")
                            final_output[k][childk] = inp[k][childk]
                        else:
                            # print("comes here")
                            final_output[k] = {}
                            final_output[k][childk] = inp[k][childk]
                else:
                    if len(inp[k]) > 0:
                        # print(inp[k])
                        if inp[k][0] != 'NA':
                            final_output[k] = inp[k]

        # print(f"final: {final_output}")
                            
    def get_needed_format(input_data, auditors_system_prompt):
        global gpt_model
        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{auditors_system_prompt}"},
                {"role": "user", "content": f"{input_data}"}
            ]
        )

        return gpt_response.choices[0].message['content']

    # @timeout_decorator.timeout(timeout)  # Set a 20-second timeout
    def run_child_llm_with_timeout(user_content, system_prompt, gpt_model, max_tokens=200):
        global timeout
        global log_str
        # print(f"timeout = {timeout}")
        # print(f"timeout = {timeout}\\n")
        log_str += f"timeout = {timeout}\\n"
        try:
            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                # model="gpt-4",
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"{user_content}"}
                ],
                max_tokens=max_tokens
            )
            return gpt_response
        except openai.error.RateLimitError as e:
            # print(f"Rate limit error in LLM. {e}")
            if ('You exceeded your current quota, please check your plan and billing details. For more information on'
                ' this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.'
                    in str(e)):
                print(e)
                sys.exit(1)
            return {"timeouterror": 0}
        except timeout_decorator.TimeoutError:
            # print("LLM timed out.")
            log_str += "LLM timed out.\n"
            return {"timeouterror": 0}
        except Exception as e:
            # print(f"Other LLM error. {e}")
            log_str += f"Other LLM error:  {e}\\n"
            return {"timeouterror": 0}

    def run_llm(user_content, system_prompt, gpt_model, max_tokens=200):
        global log_str
        run_idx = 0
        # rerun_counter = 1
        max_rerun = 3
        while run_idx < max_rerun:
            gpt_response = run_child_llm_with_timeout(
                user_content, system_prompt, gpt_model, max_tokens)
            if 'timeouterror' in gpt_response.keys():
                # print(f"Rerun count {run_idx}")
                log_str += f"Rerun count {run_idx}\\n"
                # print(f"Rerun count {run_idx}\\n")
                pass
            else:
                return gpt_response
            run_idx += 1
        # if code came here means not a single llm run was successful
        return {"timeouterror": 0}

    def run_gpt(page_text, system_prompt, max_tokens=200, start_pos=0, end_pos=doc_parts):
        global log_str
        global doc_parts  # in how many parts we divide the document
        global timeout

        previous_response_length = 0
        new_response_length = 0
        # if new response is 40% lesser than previous response then ignore it
        comparison_threshold = 0.6
        previous_response = ''
        new_response = ''
        trimmed_response_list = []
        translation_table = str.maketrans('', '', "[]\'\":}{\\")
        similarity_threshold = 0.9

        timeout = 30
        page_len = len(page_text)
        text_buffer = int(
            page_len * 0.05)  # in every iteration, start about 10% behind so as to cover any text breaks between iterations

        gpt_model = "gpt-3.5-turbo"
        # gpt_model = 'gpt-4'
        i = start_pos
        response_list = []
        while i < end_pos:
            if gpt_model == "gpt-4":
                # to overcome LLM token rate limit
                time.sleep(int(60 / doc_parts) + 2)
            else:
                time.sleep(1)

            txt_from = int(page_len / doc_parts)
            start = (txt_from * i) - text_buffer
            if start < 0:
                start = (txt_from * i)

            if i == (doc_parts - 1):
                end = page_len
            else:
                end = txt_from * (i + 1)

            # print("Start", start)
            # print("End", end)
            log_str += f"------iteration {i}\\n"
            # print(f"------iteration {i}")
            # print(f"------iteration {i}\\n")

            user_content = page_text[start: end]

            # analyse the data now
            try:
                # print(f"text length: {len(system_prompt + user_content)}")
                log_str += f"text length: {len(system_prompt + user_content)}\\n"
                log_str += f"max tokns: {max_tokens}\\n"
                # print(f"text length: {len(system_prompt + user_content)}\\n")
                # print(f"max tokns: {max_tokens}")
                # print(f"max tokns: {max_tokens}\\n")
                gpt_response = run_llm(
                    user_content, system_prompt, gpt_model, max_tokens)
                if 'timeouterror' not in gpt_response.keys():
                    tokens = gpt_response['usage']['total_tokens']
                    # print(f"-----tokens used: {tokens}-----")
                    # print(f"-----tokens used: {tokens}-----\\n")
                    log_str += f"-----tokens used: {tokens}-----\\n"
                    if gpt_model != 'gpt-4':
                        dt = gpt_response['choices'][0]['message']['content']
                        fixed_json_str = dt.replace("\'", "\"")
                        new_response_length = len(fixed_json_str)
                        new_response = fixed_json_str.translate(
                            translation_table).strip()
                        first_level_reponse = json.loads(fixed_json_str)
                    else:
                        dt = gpt_response['choices'][0]['message']['content']
                        fixed_json_str = dt.replace("\'", "")
                        new_response_length = len(fixed_json_str)
                        new_response = fixed_json_str.translate(
                            translation_table).strip()
                        first_level_reponse = json.loads(fixed_json_str)

                    # print(f"-----every answer: {new_response}")
                    log_str += f"-----every answer: {new_response}\\n"
                    # print(f"-----every answer: {new_response}\\n")

                    if (new_response_length > (previous_response_length * comparison_threshold)) & (
                            new_response not in trimmed_response_list):
                        max_similarity = 0
                        for comp_str in trimmed_response_list:
                            seq_match = difflib.SequenceMatcher(
                                None, new_response, comp_str)
                            similarity = seq_match.ratio()
                            if similarity > max_similarity:
                                max_similarity = similarity

                        # print(f"max threshold {max_similarity}")
                        # print(f"max threshold {max_similarity}\\n")
                        log_str += f"max threshold {max_similarity}\\n"

                        if max_similarity < similarity_threshold:
                            response_list.append(first_level_reponse)
                            trimmed_response_list.append(new_response)
                            # print(f"before previous: {previous_response_length} new: {new_response_length}")
                            previous_response_length = new_response_length
                            previous_response = new_response
                            # print(f"after previous: {previous_response_length} new: {new_response_length}")
                            # print("appending")
                            log_str += "appending. \n"

                            # print("appending. \n")
                        else:
                            # print("not appending")
                            log_str += "not appending.\n"
                            # print("not appending.\n")
                    else:
                        # print("not appending")
                        log_str += "not appending.\n"

                        # print("not appending.\n")
                    new_response_length = 0
                    new_response = ''
                    # print("***********")
                    log_str += "**********\n"  

                    # print("**********\n")
            except openai.error.RateLimitError as e:
                # print(f"Rate limit error in LLM. {e}")
                log_str += f"Rate limit error in LLM. {e}\\n"
                log_str += traceback.format_exc()
                time.sleep(15)
            except openai.error.InvalidRequestError as e:
                # print(f"LLM error. {e}")
                log_str += f"LLM error. {e}\\n"
            except timeout_decorator.TimeoutError:
                # Handle a timeout error
                # print("API call timed out")
                log_str += "API call timed out\n"
            except Exception as e:
                # print(f"Token limit error in LLM. {e}")
                log_str += f"Other LLM error in LLM. {e}\\n"
                log_str += traceback.format_exc()
                # print("############")
                # print(gpt_response['choices'][0]['message']['content'])
                log_str += gpt_response['choices'][0]['message']['content']
            i += 1

        answer_checker(system_prompt, response_list, max_tokens)


    # ----------------------------------------------------------------------
    basic_details_prompt = """
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about a company.
    * from this input, extract the details asked below about the company using given rules.
    * rules can be interpreted as:
    * <detail to extract> - <rules to find the correspodning detail>

    rules:
    * <company name> - name of the reported company name
    * <company type> - this can be found in company name. it can have values like 'private limited', 'limited liability company', 'proprietership' etc. this can be found in the reported company name
    * <DUN number> - this is the DUNS number and has the format xx-xxx-xxxx, where every 'x' is a digit
    * <headquarter> - this is the company's headquarter address. this can be found on the very early part of the input, or in a few other places. an address is a headquarter if its given in the very early part of the input, or if an address is labelled as either 'registered office' or 'headquarter'


    provide output in following json format:
    {
    "company name":[<company name>],
    "company type":[<company type],
    "DUN":[<DUNS>],
    "headquarter":[<headquarter address>]
    }

    if you find multiple answers to one detail, only provide the one that with the most confidence.
    if you do not have high confidence for finding any detail, respond with blank('') for that detail item
    """

    # final_output={}
    # print("Processing basic details.")
    run_gpt(page_text, basic_details_prompt, max_tokens=200, start_pos=0, end_pos=2)

    try:
        company_name = final_output['company name']
        if len(company_name) == 1:
            company_name = company_name[0]
    except:
        company_name = ''

    page_text = update_pagetext(pages, company_name)

    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", len(page_text), "+++++++++++++++++++++++++++++++++++++++++++++++")

    basic_details_prompt = """
    * the input is a business report about a company.
    * from this input, extract the details asked below about the company using given rules.
    * rules can be interpreted as:
    * <detail to extract> - <rules to find the correspodning detail>

    rules:
    * <industry> - this is the business industry this company belongs to
    * <incorporation date> - this is the incorporation date of the company
    * <Tangible Revenue> - provided following format:
    Tangible Revenue
    <tangible revenue amount with currency & denomination>
    * Tangible Net worth - provided following format:
    Tangible Net worth
    <tangible networth amount with currency & denomination>
    * Chief Executive - provided in the following format:
    Chief Executive
    <chief executive name>
    <chief executive designation>
    * <manufacturing locations> - this is the compnay's manufacturing locations.  this can be found on the very early part of the input, or in a few other places and in the form of a paragraph. a location is a manufacturing location if its given in the very early part of the input, or if an address is labelled as either 'manufacturing location' or 'factory'

    provide output in following json format:
    {
    "industry":[<industry>],
    "incorporation date":[<incorporation date>],
    "tangible revenue":[<tangible revenue amount with currency & denomination>],
    "tangible net worth":[<tangible net worth amount with currency & denomination>],
    "ceo name":[<chief executive name>],
    "ceo design":[<chief executive designation>],
    "manufacturing_locations":[<manufacturing locations>]
    }

    if you find multiple answers to one detail, only provide the one that with the most confidence.
    if you do not have high confidence for finding any detail, repond with blank('') for that detail item
    """

    # final_output={}
    print("Processing basic details.")
    run_gpt(page_text, basic_details_prompt, max_tokens=200, start_pos=0, end_pos=5)
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    sic_system_prompt = """
    * DO NOT USE any previous references/context to complete this task.
    * Your task is to extract the Standard Industry Classification (SIC) codes given in the input text ONLY
    * This information will be after strings "Standard Industry Classification (SIC) Codes", SIC Codes" and "Description"
    * A one liner description of the SIC code and extendd code will follow the SIC code
    * EXAMPLE data: 2361-0200@'Engaged in software programming'    
    Here 2361 is "SIC code", 0200 is "extended code", rest is "one liner description"
    * extract SIC information given in the input and respond in following json format:
    {
        "sic":{
            "<SIC code>-<extended code>":[<one liner description>]
        },
    }
    If you do not find SIC information in input then give following output:
    {
        "sic":['']
    }
    * before responding, think for a few seconds and check if the response is correct
    * start again if the above rules are not followed then start again
    """
    sic_system_prompt = sic_system_prompt
    # print("Processing SIC code")
    run_gpt(page_text, sic_system_prompt, max_tokens=150, start_pos=int(doc_parts / 1.5), end_pos=doc_parts)
    # -------------------------------------------------

    try:
        sic_data = final_output['sic']
        sic_prompt = ''
        if isinstance(sic_data, dict):
            for k in sic_data.keys():
                # print(k)
                if (len(sic_data[k]) > 0):
                    if (sic_data[k][0] != 'NA'):
                        sic_prompt += sic_data[k][0] + "/"

        # print("Processing SIC summary")
        if len(sic_prompt) > 10:
            # the above is an arbitrary check to ensure there is value in sic_prompt
            sic_summary_prompt = f"""
            * {sic_prompt} represents sic descriptions separated by "/"
            * summarise that in a one liner summary
            * give output in following format:
            {{
                "sic summary":[<one liner summary generated from input>]
            }}
            if the summary cannot be created, then produce:
            {{
                "sic summary":['']
            }}
            """

            user_content = ''
            gpt_response = run_llm(user_content, sic_summary_prompt, gpt_model, max_tokens=100)
            # print(gpt_response)
            if 'timeouterror' not in gpt_response.keys():
                tokens = gpt_response['usage']['total_tokens']
                # print(f"-----tokens used: {tokens}-----")

                if gpt_model != 'gpt-4':
                    dt = gpt_response['choices'][0]['message']['content']
                    # Replace single quotes with double quotes while avoiding changing single quotes that are part of the syntax
                    # print("^^^^^^^^^^^^^^")
                    fixed_json_str = dt.replace("\'", "\"")
                    first_level_reponse = json.loads(fixed_json_str)
                else:
                    dt = gpt_response['choices'][0]['message']['content']
                    # Replace single quotes with double quotes while avoiding changing single quotes that are part of the syntax
                    # print("^^^^^^^^^^^^^^")
                    fixed_json_str = dt.replace("\'", "")
                    # print(fixed_json_str)
                    first_level_reponse = json.loads(fixed_json_str)

                final_output['sic_summary'] = first_level_reponse
                # print("***********")
            else:
                final_output['sic_summary'] = ['']
    except:
        final_output['sic_summary'] = ['']
    # -------------------------------------------------

    sic_summary = final_output['sic_summary'] if 'sic_summary' in final_output.keys() else ''

    try:
        industry = final_output['industry'][0]
    except:
        industry = ''

    product_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and extract names of products this company manufactures, if provided in the input.
    * product information can be present in a table where one of the columns may be called 'product' or something simialr
    * some information about products can be found in {sic_summary}. this will contain categorisation of products, not entire product names
    * product information can also be present in following format:
    Product
    Product@<some label>
    <product name>@<other product information>
    * product infomration can also be present in text
    * Product information can be across many parts in input
    * extract names of upto 6 products names, if any, and respond in following json format:
    {{
        "products":[<product1 name>,<product2 name>,...]
    }}
    * if you're not sure, respond with the following format:
    {{
        "products":['']
    }}
    """
    # print("Processing Products.")
    log_str += "Processing Products.\n"
    product_system_prompt = product_system_prompt

    run_gpt(page_text, product_system_prompt, max_tokens=250, start_pos=2, end_pos=8)
    # -------------------------------
    # company_name=final_output['company name'][0]
    service_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and extract names of services this company offers, if provided in the input.
    * service information can be present in a table where one of the columns may be called 'service' or something simialr
    * some information about services can be found in {sic_summary}, which describes categories of all SIC codes of this company
    * service information can also be present in following format:
    * extract names of upto 6 services, if any, and respond in following json format:
    {{
        "services":[<service1 name>,<service2 name>,...]
    }}
    * if you're not sure, respond with the following format:
    {{
        "services":['']
    }}
    """
    service_system_prompt = service_system_prompt
    # print("Processing Services.")
    log_str += "Processing Services.\n"
    run_gpt(page_text, service_system_prompt, max_tokens=250, start_pos=2, end_pos=8)
    # ----------------------------------------

    # company_name=final_output['company name'][0]

    about_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * act as a financial analyst and summarise detail about the company, its history, products, services, and its activities in 3-4 sentences.
    * do not talk about banking details, financial information, capital structure, working capital etc in the summary
    * provide the output in the following json format:
    {{
        "about":[<company's detail>]
    }}
    * if you're not sure, respond with the following format:
    {{
        "about":['']
    }}
    """
    about_system_prompt = about_system_prompt
    # print("processing About Company")
    log_str += "processing About Company\n"
    run_gpt(page_text, about_system_prompt, max_tokens=200, start_pos=0, end_pos=3)
    # -----------------------------------------

    # company_name=final_output['company name'][0]
    try:
        about_info = final_output['about'][0]
    except:
        about_info = ""
    line_of_business_system_prompt = f"""
    * summarise {about_info} into 3 lines while talking about products or services of {company_name}. this becomes summary1
    * further summarise summary1 into 1 line. this becomes summary2
    * respond in following json format:
    {{
        "line of business":[<summary1>],
        "lob":[<summary2>]
    }}
    """

    user_content = ''
    gpt_response = run_llm(user_content, line_of_business_system_prompt, gpt_model, max_tokens=200)
    if 'timeouterror' not in gpt_response.keys():
        tokens = gpt_response['usage']['total_tokens']
        # print(f"-----tokens used: {tokens}-----")

        dt = gpt_response['choices'][0]['message']['content']
        # Replace single quotes with double quotes while avoiding changing single quotes that are part of the syntax
        fixed_json_str = dt.replace("\'", "\"")
        # print(f"response: {fixed_json_str}")
        try:
            first_level_reponse = json.loads(fixed_json_str)
        except:
            ### if returned value is not a json format, use the string as output
            first_level_reponse = fixed_json_str
        final_output['about company'] = first_level_reponse
        # print("***********")
    else:
        final_output['about company'] = ['']

    # print("Processing line of business")
    log_str += "Processing line of business\n"
    # ----------------------

    director_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and extract names of directors of this company, their DIN numbers and their designation
    * this information will be found between string labels 'MANAGEMENT PROFILE' and 'SHAREHOLDING PATTERN'
    * columns will be 'DIN', 'name of directors', 'designation'
    * DIN is a 8 digit number
    * every row will contain information for one director
    * extract information in following json format:
    {{
        "directors":{{
            "<name of director>":[<DIN>,<Designation>]
        }}
    }}
    * If you're not sure, respond with ''.
    * DO NOT PROVIDE IMAGINARY ANSWERS.
    """

    director_system_prompt = director_system_prompt
    # print("Processing directors")
    log_str += "Processing directors\n"
    run_gpt(page_text, director_system_prompt, max_tokens=150, start_pos=8, end_pos=15)

   
    shareholding_pattern_system_prompt = f"""
    * Do noat use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and extract shareholding pattern of this company.
    * in the input, 'shareholding pattern' data is present between string labels 'SHAREHOLDING PATTERN' and 'FINANCIAL PERFORMANCE &
    TRENDS'
    * this information is provided as: 'Shareholders name', 'Shares held', '% Held'
    * 'shareholder name' will be a person's name
    * each row will provide information about one shareholder and their percentage holding
    * extract shareholding pattern information in following json format:
    {{
        "shareholding":{{
            "<shareholder name>":[<%held>],
            ...
        }}
    }}
    * if you're not sure, respond with ''.
    """

    shareholding_pattern_system_prompt = shareholding_pattern_system_prompt
    # print("Processing shareholding")
    log_str += "Processing shareholding\n"
    run_gpt(page_text, shareholding_pattern_system_prompt, max_tokens=100, start_pos=3, end_pos=10)

    ### END OF THREAD1

    iso_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and find out about the ISO Certificates that the company may have
    * information about ISO Certificates will start with a label 'ISO Certifications' or something similar.
    * Each row will contain following columns: Type of certificate,Certificate number,Issue date, Valid up to,Certifying agency
    * extract Type of certificate,Certificate number,Issue date, Valid up to,Certifying agency for every certificate and send output in following json format:
    {{
        "iso":{{
            "<certificate number>":[<Type of certificate>,<Issue date>, <Valid up to>,<Certifying agency>]
        }}
    }}
    * If you're not sure, respond with ''.
    """

    iso_system_prompt = iso_system_prompt
    # print("Processing ISO")
    log_str += "Processing ISO\n"
    run_gpt(page_text, iso_system_prompt, max_tokens=200, start_pos=2, end_pos=8)
    # --------

    bankers_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and find out the name of banks this company works with
    * in some cases, a bank's address may be given
    * extract name of every bank with which {company_name} has a relationship, address of bank if given, in the following json format:
    {{
        "banks":{{
            "<bank name>":[<address>]
        }}
    }}
    * If you're not sure, respond with ''.
    """

    # print("Processing banker")
    log_str += "Processing banker\n"
    run_gpt(page_text, bankers_system_prompt, max_tokens=200, start_pos=int(doc_parts/1.5), end_pos=doc_parts)
    # -------------

    # -----------

    reg_and_others_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and find out the following information about the company
    a) Registration Number - it may look like 'U24110MH1989PTC050560'
    b) Name of the Auditor
    c) Annual General Meeting Date
    d) Number of employees
    * the above information will start with a label 'Registration and others' or something similar.
    * give output in following format:
    {{
    "reg_number":<Registration Number>,
    "auditor":<Name of the Auditor>,
    "agm_date":<Annual General Meeting Date>,
    
    "number_of_employees":<Number of employees>
    }}
    * If you're not sure, respond with ''.
    """
    reg_and_others_system_prompt = reg_and_others_system_prompt
    # print("processing registration")
    log_str += "processing registration\n"
    run_gpt(page_text, reg_and_others_system_prompt, max_tokens=100, start_pos=int(doc_parts/1.5), end_pos=doc_parts)
    # -----------

    contacts_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and find out contact details of the company
    * This section may contain following labels:
    a) Registered Office Address
    b) Email
    c) website
    d) phone
    e) fax
    Extract values for above labels and give output in following format:
    {{
    "registered office address":<Registered Office Address>,
    "email":<Email>,
    "website":<website>,
    "phone":[<list of phone numbers>],
    "fax":<list of fax numbers>
    }}
    * If you're not sure, respond with ''.
    """
    contacts_system_prompt = contacts_system_prompt
    # print("Processing contact")
    run_gpt(page_text, contacts_system_prompt, max_tokens=150, start_pos=int(doc_parts/1.5), end_pos=doc_parts)
    # --------


    kyc_system_prompt = f"""
    * Do not use any previous references/sessions/answers to complete this task.
    * the input is a business report about company {company_name}, whose industry segment is {industry}.
    * Your task is to act as a financial analyst and find out the KYC details of the company
    * this information will start with a label 'KYC Details' or something similar.
    * This section will contain following labels:
    a) Company PAN - examples of how PAN looks like are: AAPAR2387M, or AGRPA3426M
    b) GST Number - example of how GST number looks like is: 22AAAAA0000A1Z5
    c) IEC Code - example of how IEC code looks like is: 3102003150
    d) TAN Number - example if how TAN number looks like is: PNEP25831B
    Extract values for above labels and give output in following json format:
    {{
    "pan": <company PAN>,
    "gst":[<GST NAME><GST number>],
    "iec":<IEC code>,
    "tan":<TAN number>
    }}
    *Important: GST number may be more than one, so give output in list format with name of GST column and its number. 
    if you dont have a high confidence in your response, say ''
    """
    # print("processing KYC")
    log_str += "processing KYC\n"
    run_gpt(page_text, kyc_system_prompt, max_tokens=100, start_pos=int(doc_parts/1.5), end_pos=doc_parts)

    # print("Retunring")
    if report_type == "GPR":
        
        group_concerns_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out the company's Group Concerns, if any 
        * information about Group Concerns will contain 'group company name', 'legal structure' and 'Line of Business' of the group company
        * 'Legal Structure' can have values like 'private limited company' or 'limited liability partnership' etc
        * extract Company Name, 'Legal Structure', and give output in given json format:
        {{
            "group concerns":{{
                "<group company name>":[<legal structure>,<Line of Business>]
            }}
        }}
        * If you're not sure, respond with ''."""
        group_concerns_system_prompt = group_concerns_system_prompt
        # print("processing group concerns")
        run_gpt(page_text, group_concerns_system_prompt, max_tokens=250, start_pos=int(doc_parts/2), end_pos=doc_parts)
            
        branch_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find information about the company's branches
        * information about branches is found between string label 'LOCATION & GROUP DETAILS' and 'OTHER INFORMATION'
        * company branches have an address, a 'location type' with values like 'godown', 'office', 'manufacturing unit' etc, and some other information 
        * extract values for every company branch and give output in following format:
        {{
        "branches":[
        [<address>,<location type>]
        ]
        }}
        * If you're not sure, respond with ''."""
        branch_system_prompt = branch_system_prompt
        # print("Processing branch")
        log_str += "Processing branch\n"
        run_gpt(page_text, branch_system_prompt, max_tokens=200, start_pos=int(doc_parts/1.5), end_pos=doc_parts)
        # -------------

        subsidiary_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out details of the company's subsidiaries
        * information about subsidiaries is found between string label 'LOCATION & GROUP DETAILS' and 'OTHER INFORMATION'
        * subsidiaries have a name, address, % shares held, and total shares held
        * extract values for every subsidiary and give output in following format:
        {{
        "subsidiary":[
        [<name>,<address>,<% shares held>]
        ]
        }}
        * Answer only when confidence is high. if subsdiary information as per above rules is not found then give the following json:
        {{
            "subsidiary":['']
        }}
        """
        # print("Processing subsidiary")
        log_str += "Processing subsidiary\n"
        run_gpt(page_text, subsidiary_system_prompt, max_tokens=100, start_pos=int(doc_parts/1.5), end_pos=doc_parts)
        # --------

        affiliate_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out details about the company's affiliates
        * information about affiliates is found between string label 'LOCATION & GROUP DETAILS' and 'OTHER INFORMATION'
        * affiliates have a name, address
        * extract values for every affiliates and give output in following format:
        {{
        "affiliates":[
            [<name>,<address>]
        ]
        }}
        * Answer only when confidence is high. if affiliates information as per above rules is not found then give the following json:
        {{
            "affiliates":['']
        }}
        """
        # print("Processing affiliate")
        log_str += "Processing affiliate\n"
        run_gpt(page_text, affiliate_system_prompt, max_tokens=100, start_pos=int(doc_parts/1.5), end_pos=doc_parts)
        # --------


        revenue_profile_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out revenue generation breakup of this company as per geography, ie how much percentage revenue comes from domestic market, and how much revenue  percentage comes from internatioanl markets
        * in the input, this info may be given in the folowing format:
        CUSTOMERS & VENDORS
        REVENUE DETAILS
        LOCAL    <% revenue from local>
        INTERNATIONAL    <% revenue from international>
        * at times either LOCAL or INTERNATIONAL values may be missing
        * extract information in following json format:
        {{
            "revenue":{{
                "local":[<% revenue from LOCAL>],
                "international":[<% revenue from International>]
            }}
        }}
        * If you're not sure, then respond with ''.
        """
        revenue_profile_system_prompt = revenue_profile_system_prompt
        # print("Processing revenue")
        log_str += "Processing revenue\n"
        run_gpt(page_text, revenue_profile_system_prompt,
                max_tokens=50, start_pos=2, end_pos=8)
        

        
        purchase_profile_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out vendor purchase breakup of this company as per geography, ie how much percentage purchase comes from domestic market, and how much purchase  percentage happens from internatioanl markets
        * in the input, this info may be given in the folowing format:
        PURCHASE DETAILS
        LOCAL    <% purchase from local>
        INTERNATIONAL    <% purchase from international>
        * at times either LOCAL or INTERNATIONAL values may be missing
        * similar looking details will be found under 'REVENUE DETAILS' section. do not consider that
        * extract information in following json format:
        {{
            "purchase":{{
                "local":[<% of purchase from local>],
                "international":[<% of purchase from international>]
            }}
        }}
        * If you're not sure, then respond with ''.
        """
        purchase_profile_system_prompt = purchase_profile_system_prompt
        # print("Processing purchase")
        log_str += "Processing purchase\n"
        run_gpt(page_text, purchase_profile_system_prompt, max_tokens=50, start_pos=2, end_pos=8)

        top_customer_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out customer names, country of customers, % of revenue from that customer (this may be with or without a % sign), length of relationship with that customer
        * customer name will look like the name of a business, not of people
        * customer name and % revenue for that customer will be mentioned close to each other in input
        * this info wll be given in following format:
        CUSTOMERS
        Name of the Customers@Country@% of total revenue@Length of@relationship (In years)
        <customer name>@<country>@<% revenue from this customer>@<years>
        * we need to identify the right location of customer information and extract the information  in following json format:
        {{
            "customers":{{
                "<customer name>":[<% of revenue from this customer>]
            }}
        }}
        * If you're not sure, respond with ''.
        * DO NOT PROVIDE IMAGINARY ANSWERS.
        """
        top_customer_system_prompt = top_customer_system_prompt

        # print("Processing customers")
        log_str += "Processing customers\n"
        run_gpt(page_text, top_customer_system_prompt, max_tokens=200, start_pos=2, end_pos=8)

        
        top_vendor_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out vendor names, country of vendor, % of purchase from that vendor (this may be with or without a % sign), length of relationship with that vendor
        * vendor name will look like the name of a business
        * vendor name and % purchase for that customer will be mentioned close to each other in input
        * this info wll be given in following format:
        VENDORS
        Name of the Vendors@Country@% of total Purchase@Length of@relationship (In years)
        <customer name>@<country>@<% Purchase from this vendor>@<years>
        * we need to identify the right location of vendor information and extract the information  in following json format:
        {{
            "vendors":{{
                "<vendor name>":[<% Purchase from this vendor>]
            }}
        }}
        * If you're not sure, respond with ''."""
        top_vendor_system_prompt = top_vendor_system_prompt
        # print("Processing vendors")
        log_str += "Processing vendors\n"
        run_gpt(page_text, top_vendor_system_prompt, max_tokens=200, start_pos=2, end_pos=8)
        # -------------------
                
        import_export_of_comp_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and find out the import and export that are done by the company, with other countries.
        * in the input, this info may be given in the folowing format:
        IMPORT MARKET or Country of Imports or Imports From
        <LIST OF COUNTRIES THE COMPANY IMPORTS FROM>
        EXPORT MARKET or Country of Exports or Exports To
        <LIST OF COUNTRIES THE COMPANY EXPORTS TO>
        * at sometimes there may not be any import or export information so there will be no value
        * if you dont find any import or export information, then respond dont add any element to the list
        * if you dont get data for any country, then dont add that country to the output list
        * make sure you add only country names dont add any other information
        * the output should be in following json format:
        {{
            "import_countries":[<list of countries the company imports from>],
            "export_countries":[<list of countries the company exports to>]
        }}
        * If you're not sure, then respond with ''.
        """
        import_export_of_comp_system_prompt = import_export_of_comp_system_prompt
        # print("Processing purchase")
        log_str += "Processing purchase\n"
        run_gpt(page_text, import_export_of_comp_system_prompt, max_tokens=150, start_pos=2, end_pos=8)
    # -------------------
    # ----------------
    elif report_type == "BPR":
        top_customer_system_prompt = f"""The input talks about the customers and vendors of the company.
        * Your task is to act as a financial analyst and find out only customer names, country of customers, % of revenue from that customer (this may be with or without a % sign), length of relationship with that customer
        * we need to identify the right location of customer information and extract the information  in following json format:
        {{
            "customers":{{
                "<customer name>":[<country>,<% of revenue from this customer>,<length of relationship>]
            }}
        }}
        *Important: Do not use vendors data or any other data that is not related to customers.
        * If you're not sure, respond with ''.
        * DO NOT PROVIDE IMAGINARY ANSWERS.
        """
        top_vendor_system_prompt = f"""
        The input talks about the customers and vendors of the company.
        * Your task is to analyze the input thoroughly and only look for vendors section and act as a financial analyst and find out only vendors names, country of customers, % of revenue from that customer (this may be with or without a % sign), length of relationship with that customer
        * we need to identify the right location of customer information and extract the information  in following json format:
        {{
            "vendors":{{
                "<vendor name>":[<country>,<% of revenue from this customer>,<length of relationship>]
            }}
        }}
        *Important: Do not use customers data or any other data that is not related to vendors.
        * If you're not sure, respond with ''.
        * DO NOT PROVIDE IMAGINARY ANSWERS.
        """

        revenue_profile_system_prompt = f"""
        The input talks about the customers and vendors of the company.
        * Your task is to analyze the input thoroughly and only look for revenue details section and act as a financial analyst and find out how much percentage of revenue comes from domestic market, and how much revenue  percentage comes from internatioanl markets
        * in the input, this information may be given in the revenue details section:
        REVENUE DETAILS
        LOCAL    <% revenue from local>
        INTERNATIONAL    <% revenue from international>
        * extract information in following json format:
        {{
            "revenue":{{
                "local":[<% revenue from LOCAL>],
                "international":[<% revenue from International>]
            }}
        }}
        *Important: Do not use customers data or vendors data or any other data that is not related to revenue details.
        *Important: At times either LOCAL or INTERNATIONAL values may be missing so just provide the available value and leave missing values.
        * If you're not sure, then respond with ''.
        """
        purchase_profile_system_prompt = f"""
        The input talks about the customers and vendors of the company.
        * Your task is to analyze the input thoroughly and only look for purchase details section and act as a financial analyst and find out how much percentage of revenue comes from domestic purchase, and how much revenue of percentage comes from internatioanl purchase.
        * in the input, this information may be given in the purchase details section:
        PURCHASE DETAILS
        LOCAL    <% revenue from local>
        INTERNATIONAL    <% revenue from international>
        * extract information in following json format:
        {{
            "purchase":{{
                "local":[<% of purchase from local>],
                "international":[<% of purchase from international>]
            }}
        }}
        *Important: Do not use customers data or vendors data or any other data that is not related to purchase details.
        *Important: At times either LOCAL or INTERNATIONAL values may be missing so just provide the available value and leave missing values.
        * If you're not sure, then respond with ''.
        """

        import_export_of_comp_system_prompt = f"""The input talks about the customers and vendors of the company.
        * Your task is to analyze the input thoroughly and only look for import and export countries and act as a financial analyst and find out the import and export that are done by the company, with other countries with their %.
        *Important: if you dont find any import or export information, then dont add any element to the list
        *Important: if you dont get data for any country, then dont add that country to the output list
        *Important: make sure you add only country names and % of import/exports, dont add any other information
        * the output should be in following json format:
        {{
            "import_countries":[<list of countries the company imports from>,<%>],
            "export_countries":[<list of countries the company exports to>,<%>]
        }}
        *Important: Dont use customers data or vendors data or any other data that is not related to import and export.
        *Important: at sometimes there may not be any import or export information so there will be no value hence return ''
        * If you're not sure, then respond with ''.
        """
        group_concerns_system_prompt = f"""The input talks about the location and group details of the company.
        * Your task is to analyze the input thoroughly and then act as a financial analyst and find out the company's Group Concerns, if any 
        * information about Group Concerns will contain 'group company name', 'legal structure' and 'Line/Nature of Business' of the group company
        * 'Legal Structure' can have values like 'private limited company' or 'limited liability partnership' etc
        * extract Company Name, 'Legal Structure', and give output in given json format:
        {{
            "group concerns":{{
                "<group company name>":[<legal structure>,<Line/Nature of Business>]
            }}
        }}
        *Important: Do not use any other data that is not related to group concerns.
        *Important: at sometimes there may be some missing values so there will be no value hence return ''
        * If you're not sure, respond with ''
        """

        branch_system_prompt = f"""
        The input talks about the location and group details of the company.
        * Your task is to analyze the input thoroughly and then act as a financial analyst and extract all information regarding the company's branches.
        * company branches have an address, a 'location type' with values like 'godown', 'office', 'manufacturing unit' etc, and some other information 
        * extract values for every company branch and give output in following format:
        {{
        "branches":[
        [<address>,<location type>]
        ]
        }}
        *Important: Do not use any other data that is not related to branches.
        *Important: at sometimes there may be some missing values so there will be no value hence return ''
        * If you're not sure, respond with ''.
        """

        ultimate_holding_prompt = f"""
        The input talks about the location and group details of the company.
        * Your task is to analyze the input thoroughly and then act as a financial analyst and extract all information regarding the company's ultimate holding companies.
        * ultimate holding company have a name of the company,and its country 
        * extract values for every ultimate holding company and give output in following format:
        {{
            "ultimate holding company":[
                [<name of the company>,<country>]
            ]
        }}
        *Important: Do not use any other data that is not related to ultimate holding companies.
        *Important: at sometimes there may be some missing values so there will be no value hence return ''
        * If you're not sure, respond with ''.
        """
        holding_company_prompt = f"""
        The input talks about the location and group details of the company.
        * Your task is to analyze the input thoroughly and then act as a financial analyst and extract all information regarding the company's holding companies.
        * holding company have a name of the company,and its country 
        * extract values for every holding company and give output in following format:
        {{
            "holding company":[
                [<name of the company>,<country>]
            ]
        }}
        *Important: Do not use any other data that is not related to holding companies.
        *Important: at sometimes there may be some missing values so there will be no value hence return ''
        * If you're not sure, respond with ''.
        """

        subsidiary_system_prompt = f"""
        The input talks about the location and group details of the company.
        Your task is to analyze the input thoroughly and then act as a financial analyst and extract all information regarding the company's subsidiaries.
        * subsidiaries have a name, address, % shares held, and total shares held
        * extract values for every subsidiary and give output in following format:
        {{
        "subsidiary":[
        [<name>,<address>,<% shares held>]
        ]
        }}
        * Answer only when confidence is high. if subsdiary information as per above rules is not found then give the following json:
        {{
            "subsidiary":['']
        }}
        *Important: Do not use any other data that is not related to subsidiaries.
        *Important: at sometimes there may be some missing values so there will be no value hence return ''
        *Important: If you're not sure, respond with ''.
        *Important: If not found, respond with ''
        *Important: DO NOT PROVIDE IMAGINARY ANSWERS.
        """

        affiliate_system_prompt = f"""
        The input talks about the location and group details of the company.
        Your task is to analyze the input thoroughly and then look for affiliates section if found then extract all information regarding the company's affiliates else return ''.
        * extract values for every affiliates and give output in following format:
        {{
        "affiliates":[
            [<name>,<address>]
        ]
        }}
        *Important: If affiliates not found, then respond with ''
        *Very Important: Do not use subsidiaries data or any other data that is not related to affiliates.
        *Important: Do not use any other data that is not related to affiliates.
        *Important: If you're not sure, respond with ''.
        *Important: DO NOT PROVIDE IMAGINARY ANSWERS.
        """
        revenue_terms_system_prompt = f"""
        The input contains information about the company's revenue terms. Your task is to look for the local revenue terms and export terms of this company. The local revenue terms will contain the number of days for open account and the export terms will contain the export terms. Extract the information in the following json format:
            * at times either LOCAL REVENUE TERMS or EXPORT TERMS values may be missing
            * extract information in following json format:
            {{
                "revenue_terms":{{
                    "local_revenue_terms": <Number of days>,
                    "export_terms":<export terms>
                }}
            }}
            * If you're not sure, then respond with ''.
            """

        purchase_terms_system_prompt = f"""
            The input contains information about the company's purchase terms. Your task is to look for the local purchase terms and import terms of this company. The local purchase terms will contain the number of days for open account and the import terms will contain the import terms. Extract the information in the following json format:
                * at times either LOCAL PURCHASE TERMS or IMPORT TERMS values may be missing
                * extract information in following json format:
                {{
                    "purchase_terms":{{
                        "local_purchase_terms":<Number of days>,
                        "import_terms":<import terms>
                    }}
                }}
                * If you're not sure, then respond with ''.
                """
        try:

            """Processing customers and related data"""

            i, j = extract_sections_with_page_numbers(pages[pg_num_content], "customers")
            customers_data = get_component_wise_data(i, j, pages)
            gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{top_customer_system_prompt}"},
                {"role": "user", "content": f"{customers_data}"}
            ]
            )

            try:
                final_output.update(json.loads(gpt_response.choices[0].message['content']))
            except Exception as e:
                # raise e
                final_output.update(eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{top_vendor_system_prompt}"},
                    {"role": "user", "content": f"{customers_data}"}
                ]
            )

            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content']))
                # print("OK")
            except Exception as e:

                # raise e
                final_output.update(eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{revenue_terms_system_prompt}"},
                    {"role": "user", "content": f"{customers_data}"}
                ]
            )

            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
            except Exception as e:
                # raise e
                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))


            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{revenue_profile_system_prompt}"},
                    {"role": "user", "content": f"{customers_data}"}
                ]
            )

            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content']))
                # print("OK")
            except Exception as e:
                # raise e
                final_output.update(eval(gpt_response.choices[0].message['content'].replace('json', '')))


            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{purchase_profile_system_prompt}"},
                    {"role": "user", "content": f"{customers_data}"}
                ]
            )

            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content']))
                # print("OK")
            except Exception as e:

                final_output.update(eval(gpt_response.choices[0].message['content'].replace('json', '')))


            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{import_export_of_comp_system_prompt}"},
                    {"role": "user", "content": f"{customers_data}"}
                ]
            )
            
            try:
                final_output.update(json.loads(gpt_response.choices[0].message['content']))
                # print("OK")
            except Exception as e:
                final_output.update(eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{purchase_terms_system_prompt}"},
                    {"role": "user", "content": f"{customers_data}"}
                ]
            )
            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
                # print("OK")
            except Exception as e:
                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))
                
        except Exception as e:

            """Following old method if got any error"""

            revenue_profile_system_prompt = revenue_profile_system_prompt
            # print("Processing revenue")
            log_str += "Processing revenue\n"
            run_gpt(page_text, revenue_profile_system_prompt,
                    max_tokens=50, start_pos=2, end_pos=8)
            
            purchase_profile_system_prompt = purchase_profile_system_prompt
            # print("Processing purchase")
            log_str += "Processing purchase\n"
            run_gpt(page_text, purchase_profile_system_prompt,
                    max_tokens=50, start_pos=2, end_pos=8)
            
            import_export_of_comp_system_prompt = import_export_of_comp_system_prompt
            # print("Processing purchase")
            log_str += "Processing purchase\n"
            run_gpt(page_text, import_export_of_comp_system_prompt,
                    max_tokens=150, start_pos=2, end_pos=8)
            

            top_customer_system_prompt = top_customer_system_prompt

            # print("Processing customers")
            log_str += "Processing customers\n"
            run_gpt(page_text, top_customer_system_prompt,
                    max_tokens=200, start_pos=2, end_pos=8)


            top_vendor_system_prompt = top_vendor_system_prompt
            # print("Processing vendors")
            log_str += "Processing vendors\n"
            run_gpt(page_text, top_vendor_system_prompt,
                    max_tokens=200, start_pos=2, end_pos=8)
            
            revenue_terms_system_prompt = revenue_terms_system_prompt
            # print("Processing revenue terms")

            log_str += "Processing revenue\n"
            run_gpt(page_text, revenue_terms_system_prompt,
                    max_tokens=50, start_pos=2, end_pos=8)

            purchase_terms_system_prompt = purchase_terms_system_prompt
            # print("Processing revenue terms")

            log_str += "Processing revenue\n"
            run_gpt(page_text, purchase_terms_system_prompt,
                    max_tokens=50, start_pos=2, end_pos=8)
            
        try:
            """Processing locations and related data"""

            i, j = extract_sections_with_page_numbers(
                pages[pg_num_content], "location")
            location_data = get_component_wise_data(i, j, pages)
            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{group_concerns_system_prompt}"},
                    {"role": "user", "content": f"{location_data}"}
                ]
            )
            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
            except Exception as e:
                # print(gpt_response.choices[0].message['content'])
                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{branch_system_prompt}"},
                    {"role": "user", "content": f"{location_data}"}
                ]
            )
            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
                # print("OK --> Branch")
            except Exception as e:
                # print(gpt_response.choices[0].message['content'])

                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{ultimate_holding_prompt}"},
                    {"role": "user", "content": f"{location_data}"}
                ]
            )
            try:
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
            except Exception as e:
                # print(gpt_response.choices[0].message['content'])

                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{holding_company_prompt}"},
                    {"role": "user", "content": f"{location_data}"}
                ]
            )

            try:
                # print(gpt_response.choices[0].message['content'])
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
                # print("OK  --> Holding")
            except Exception as e:
                # print(gpt_response.choices[0].message['content'])

                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{subsidiary_system_prompt}"},
                    {"role": "user", "content": f"{location_data}"}
                ]
            )

            try:
                # print(gpt_response.choices[0].message['content'])
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
                # print("OK  --> Subsidiary")
            except Exception as e:
                # print(gpt_response.choices[0].message['content'])

                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{affiliate_system_prompt}"},
                    {"role": "user", "content": f"{location_data}"}
                ]
            )

            try:
                # print(gpt_response.choices[0].message['content'])
                final_output.update(json.loads(
                    gpt_response.choices[0].message['content'].replace('json', '')))
                # print("OK --> Affiliate")
            except Exception as e:
                # print(gpt_response.choices[0].message['content'])

                final_output.update(
                    eval(gpt_response.choices[0].message['content'].replace('json', '')))

        except Exception as e:
            group_concerns_system_prompt = group_concerns_system_prompt
            # print("processing group concerns")
            run_gpt(page_text, group_concerns_system_prompt, max_tokens=250,
                    start_pos=int(doc_parts/2), end_pos=doc_parts)
            
            branch_system_prompt = branch_system_prompt
            # print("Processing branch")
            # log_str += "Processing branch\n"
            run_gpt(page_text, branch_system_prompt, max_tokens=200, start_pos=int(doc_parts/1.5), end_pos=doc_parts)

            # print("Processing subsidiary")
            # log_str += "Processing subsidiary\n"
            run_gpt(page_text, subsidiary_system_prompt, max_tokens=100, start_pos=int(doc_parts/1.5), end_pos=doc_parts)

            log_str += "Processing affiliate\n"
            run_gpt(page_text, affiliate_system_prompt, max_tokens=100, start_pos=int(doc_parts/1.5), end_pos=doc_parts)

            # print("Processing ultimate holding")
            log_str += "Processing branch\n"
            run_gpt(page_text, ultimate_holding_prompt, max_tokens=200,
                    start_pos=int(doc_parts/3), end_pos=int(doc_parts/2))
            
            # print("Processing holding")
            log_str += "Processing branch\n"
            run_gpt(page_text, holding_company_prompt,
                    max_tokens=200, start_pos=7, end_pos=12)
    # -------------

    """
    Processing Auditors information from old report
    """

    i, j = extract_sections_with_page_numbers(pages[pg_num_content], "auditor")

    auditors_prompt = """
        The input data contains the information of auditors for a company. Your task is to look and understand the input clearely and extract all the text and all the tables from the input making sure that the data is in the same order as it is in the input. The input contains the data of the company for a particular year.
        *Important: Give the output in the following json format.
        {
        "year":"",
        "data": [
            'text1' : ,
            'table1':[{
                    'columns': column_names,
                    'data': values
                }],
            'text2' :,
            'table2':[{
                    'columns': column_names,
                    'data': values
                }],
        ]
        }
        *Very Important: All the tables must be extracted and in the same order as they are in the input.
        *Very Important: Do not use data that is not present in the input.
        """
    output1 = get_audit_result(i, j, pages=pages)
    if output1:
        individual_dicts = [{key: value}
                            for key, value in output1.items()]
        individual_dicts[-1][list(individual_dicts[-1].keys())[0]] = list(
            individual_dicts[-1].values())[0].split("Contingent Liabilit")[0]
        output_1 = get_needed_format(individual_dicts[0], auditors_prompt)
        try:
            individual_dicts[0] = json.loads(output_1.replace('json', ''))
            year = output1['year']
        except:
            year = list(individual_dicts[0].keys())[0]

        s3_file_name_1 = f'files/{comp_id}/{report_id}/data_from_auditor_report_{year}.json'

        writetos3(aws_access_key_id, aws_secret_access_key,
                  individual_dicts[0], s3_file_name_1)
        try:
            output_2 = get_needed_format(individual_dicts[1], auditors_prompt)
            try:
                individual_dicts[1] = json.loads(output_2.replace('json', ''))
                year = output_2['year']
            except:
                year = list(individual_dicts[1].keys())[0]
            s3_file_name_2 = f'files/{comp_id}/{report_id}/data_from_auditor_report_{year}.json'
            # print(individual_dicts[1])
            writetos3(aws_access_key_id, aws_secret_access_key,
                      individual_dicts[1], s3_file_name_2)
        except:
            pass
    else:
        pass


    if report_type == "BPR":

        solution_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and extract names of solutions this company manufactures, if provided in the input.
        * solution information can be present in a table where one of the columns may be called 'solution' or something simialr
        * some information about solutions can be found in {sic_summary}. this will contain categorisation of solutions, not entire solution names
        * solution information can also be present in following format:
        solution
        solution@<some label>
        <solution name>@<other solution information>
        * solution infomration can also be present in text
        * solution information can be across many parts in input
        * extract names of upto 6 solutions names, if any, and respond in following json format:
        {{
            "solutions":[<solution1 name>,<solution2 name>,...]
        }}
        * if you're not sure or not found then, respond with the following format:
        {{
            "solutions":['']
        }}
        """
        # print("Processing Solutions.")
        run_gpt(page_text, solution_system_prompt,
                max_tokens=250, start_pos=2, end_pos=8)

        components_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and extract names of components of this company, if provided in the input.
        * components information can be present in a table where one of the columns may be called 'components' or something similar
        * some information about 'components' can be found in {sic_summary}. this will contain categorisation of components, not entire component names
        * component information can also be present in following format:
        component
        component@<some label>
        <component name>@<other component information>
        * component infomration can also be present in text
        * component information can be across many parts in input
        * extract names of upto 6 components names, if any, and respond in following json format:
        {{
            "components":[<component1 name>,<component2 name>,...]
        }}
        * if you're not sure or not found then, respond with the following format:
        {{
            "components":['']
        }}
        *Important: Do not use data that is not present in the input.
        *Important: If not found, then respond with above format.
        """
        # print("Processing components.")
        run_gpt(page_text, components_system_prompt,
                max_tokens=250, start_pos=2, end_pos=8)

        key_executive_system_prompt = f"""
        * Do not use any previous references/sessions/answers to complete this task.
        * the input is a business report about company {company_name}, whose industry segment is {industry}.
        * Your task is to act as a financial analyst and extract names of executives of this company, and their current title
        * this information will be found between string labels 'MANAGEMENT PROFILE' and 'SHAREHOLDING PATTERN'
        * columns will be 'name of executive', 'current title'
        * every row will contain information for one executive
        * extract information in following json format:
        {{
            "executives":{{
                "<name of executive>":[<current title>]
            }}
        }}
        * If you're not sure, respond with ''.
        * DO NOT PROVIDE IMAGINARY ANSWERS.
        """

        # print("Processing key executives")
        log_str += "Processing directors\n"
        run_gpt(page_text, key_executive_system_prompt,
                max_tokens=250, start_pos=5, end_pos=10)

    """
    Saving to s3
    comp_id and report_id are passed locally for testing
    """
    s3_file_name = f'files/{comp_id}/{report_id}/data_from_oldreport.json'
    log_str += f"final out---- {final_output}"
    log_str += writetos3(aws_access_key_id, aws_secret_access_key, final_output, s3_file_name)
    try:
        with open(log_path, 'w') as file:
            file.write(log_str)    
    except:
        pass
    return final_output

# pdfname = "" 


# report = getOldReport(comp_id="183075073", report_id="1274741772",file_name="")
# print(report)
function_name = sys.argv[1]


if (function_name == "getOldReport"):
    report = getOldReport(
        pdf_path=sys.argv[2], comp_id=sys.argv[3], report_id=sys.argv[4], report_type = sys.argv[5])
    print(report)

"""
Reading the pdf from s3 
"""