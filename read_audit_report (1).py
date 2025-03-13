import os
import fitz  # PyMuPDF
import numpy as np
import shutil
import pytesseract
from json.decoder import JSONDecodeError
from PIL import Image
import json
import boto3
from botocore.exceptions import ClientError
import traceback
import json
import os
import traceback
from io import BytesIO
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import boto3
import openai
import sys
import pandas as pd
import time
import boto3
import re
import datetime
import pandas as pd
import os
import time
# from secrets_manager import get_secret
from dotenv import load_dotenv

load_dotenv()
# secret = get_secret()./
secret = ["", "", "", "snuckworks", "ap-south-1"]

# s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id,
#                   aws_secret_access_key=aws_secret_access_key)

final_output = {}

# gpt_model = "gpt-3.5-turbo"
gpt_model = os.getenv("FINETUNED_MODEL_ID_1")


aws_access_key_id = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_KEY")

openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")


s3 = boto3.client("s3", region_name=secret[4])


def get_pdf_tables(pdf_path, comp_name):
    s3_bucket_name = secret[3]
    region = "ap-south-1"
    dataframes = []
    summary_lst = []

    def start_job(client, s3_bucket_name, object_name):
        response = client.start_document_analysis(
            DocumentLocation={
                "S3Object": {"Bucket": s3_bucket_name, "Name": object_name}
            },
            FeatureTypes=["TABLES"],
        )

        job_id = response["JobId"]
        return job_id

    def is_job_complete(client, job_id):
        time.sleep(1)
        response = client.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]

        while status == "IN_PROGRESS":
            time.sleep(1)
            response = client.get_document_analysis(JobId=job_id)
            status = response["JobStatus"]
            continue

        return status

    def get_job_results(client, job_id):
        pages = []
        time.sleep(1)
        response = client.get_document_analysis(JobId=job_id)
        pages.append(response)

        next_token = None
        if "NextToken" in response:
            next_token = response["NextToken"]

        while next_token:
            time.sleep(1)
            response = client.get_document_analysis(
                JobId=job_id, NextToken=next_token)
            pages.append(response)
            next_token = None
            if "NextToken" in response:
                next_token = response["NextToken"]

        return pages

    def get_relationship_ids(block, type):
        for rels in block.get('Relationships', []):
            if rels['Type'] == type:
                yield from rels['Ids']

    def map_blocks(blocks, block_type):
        return {
            block["Id"]: block for block in blocks if block["BlockType"] == block_type
        }

    def convert_to_df(tables, cells, words, selections, merged_cells, titles):
        for table in tables.values():
            try:
                table_summary_lst = []

                # print(table)
                # Determine all the cells that belong to this table
                table_cells = [cells[cell_id]
                               for cell_id in get_relationship_ids(table, 'CHILD')]

                # Determine the table's number of rows and columns
                n_rows = max(cell['RowIndex'] for cell in table_cells)
                n_cols = max(cell['ColumnIndex'] for cell in table_cells)
                content = [[None for _ in range(n_cols)]
                           for _ in range(n_rows)]

                # get cells of table
                for cell in table_cells:
                    cell_contents = []

                    table_summary = False
                    if "EntityTypes" in cell and 'TABLE_SUMMARY' in cell['EntityTypes']:
                        table_summary = True

                    for child_id in get_relationship_ids(cell, 'CHILD'):
                        if child_id in words:
                            cell_contents.append(words[child_id]['Text'])
                        elif child_id in selections:
                            cell_contents.append(
                                selections[child_id]['SelectionStatus'])

                    i = cell['RowIndex'] - 1
                    j = cell['ColumnIndex'] - 1

                    cell_text = ' '.join(cell_contents)

                    if (table_summary):
                        is_number = bool(cell_text) and ''.join(c for c in cell_text if c.isdigit(
                        ) or c == '-').lstrip('-').replace('.', '', 1).isdigit()
                        if not is_number and cell_text:
                            table_summary_lst.append(cell_text)

                    content[i][j] = cell_text

                # get title of table
                title_text = ''
                for title_id in get_relationship_ids(table, 'TABLE_TITLE'):
                    title_contents = []
                    if title_id in titles:
                        for cell_id in get_relationship_ids(titles[title_id], 'CHILD'):
                            if cell_id in words:
                                title_contents.append(words[cell_id]['Text'])
                        title_text = ' '.join(title_contents)

                        if (title_text):
                            title_text = title_text.upper()
                        if comp_name and comp_name in title_text:
                            title_text = title_text.replace(
                                comp_name.upper(), "")

                            # get merged cells
                table_merged_cells = []
                try:
                    for cell_id in get_relationship_ids(table, 'MERGED_CELL'):
                        if 'EntityTypes' in merged_cells[cell_id] and 'COLUMN_HEADER' in merged_cells[cell_id]['EntityTypes'] and merged_cells[cell_id]['RowIndex'] == 1:

                            table_merged_cells.append({"RowIndex": merged_cells[cell_id]['RowIndex'], "ColumnIndex": merged_cells[cell_id]['ColumnIndex'],
                                                       "RowSpan": merged_cells[cell_id]['RowSpan'], "ColumnSpan": merged_cells[cell_id]['ColumnSpan']})
                except Exception as e:
                    pass

                if (len(table_merged_cells)):
                    rowSpan = table_merged_cells[0]['RowSpan']
                    restructured_column = restructure_merge_column(
                        table_merged_cells, content[:rowSpan])
                    try:
                        dataframe = pd.DataFrame(
                            content[rowSpan:], columns=restructured_column)
                        # print(tabulate(dataframe, headers='keys', tablefmt='psql'))
                    except Exception as e:
                        # print("ERROR", e)
                        dataframe = pd.DataFrame(
                            content[1:], columns=content[0])
                        # print(tabulate(dataframe, headers='keys', tablefmt='psql'))

                    summary_lst.append(table_summary_lst)
                    dataframe.name = title_text
                    dataframes.append(dataframe)

                else:
                    # We assume that the first row corresponds to the column names
                    dataframe = pd.DataFrame(content[1:], columns=content[0])
                    # print(tabulate(dataframe, headers='keys', tablefmt='psql'))

                    summary_lst.append(table_summary_lst)
                    dataframe.name = title_text
                    dataframes.append(dataframe)

            except Exception as e:
                # print(e)
                continue

    client = boto3.client("textract", region_name=region)
    job_id = start_job(client, s3_bucket_name, pdf_path)

    if is_job_complete(client, job_id):
        pages = get_job_results(client, job_id)
        # print("============  pages  ============")
        # print(pages)
        # save_to_file(json.dumps(pages), "../temp/vyjayanti.json")
        # print("============  pages  ============")

        # pages = []
        # with open("../temp/access_design.json", 'r') as file:
        #     pages = json.load(file)

        for i in range(pages[0]['DocumentMetadata']['Pages']):
            filtered_blocks = [
                block for item in pages for block in item['Blocks'] if block['Page'] == i + 1]
            blocks = filtered_blocks

            tables = map_blocks(blocks, "TABLE")
            cells = map_blocks(blocks, "CELL")
            words = map_blocks(blocks, "WORD")
            selections = map_blocks(blocks, "SELECTION_ELEMENT")
            merged_cells = map_blocks(blocks, 'MERGED_CELL')
            titles = map_blocks(blocks, 'TABLE_TITLE')

            convert_to_df(tables, cells, words,
                          selections, merged_cells, titles)

    return {"dataframes": dataframes, "summary_lst": summary_lst}


def restructure_merge_column(column_info, column1):
    # print("COLUMN INFO",column_info)
    # print("COLUMN",column1)

    restructured_column = []

    for info in column_info:
        row_index = info['RowIndex'] - 1
        col_index = info['ColumnIndex'] - 1
        row_span = info['RowSpan']
        col_span = info['ColumnSpan']

        # If ColumnSpan is more than 1, concatenate the strings
        if col_span > 1:
            col_name = ''
            for i in range(col_span):
                col_name += ''.join(column1[0][col_index + i] + " ")

            if (row_span > 1):
                for i in range(row_span):
                    col_name += column1[i][col_index] + " "

            for i in range(col_span):
                col_name__ = ""
                for j in range(len(column1)):
                    if j > row_span - 1:
                        col_name__ += ''.join(column1[j][col_index + i] + " ")
                restructured_column.append(col_name + col_name__)

        else:
            col_name = ''
            for i in range(row_span):
                col_name += column1[i][col_index] + " "
            restructured_column.append(col_name)
    return restructured_column


def pdf_to_images(pdf_path, output_folder, image_format='png', dpi=300):
    # Create a folder named "images" if it doesn't exist

    pdf_document = fitz.open(pdf_path)
    length = len(pdf_document)
    # print("Length", length)
    for page_number in range(length):
        # for page_number in range(20):
        # Get the page
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))

        # Save the image inside the "images" folder
        image_path = os.path.join(
            output_folder, f"page_{page_number + 1}.{image_format}")
        pix.save(image_path)

        # print(f"Page {page_number + 1} saved as {image_path}")

    pdf_document.close()

    return length


def writetos3(data_to_write, output_path):
    region = secret[4]
    bucket = secret[3]

    try:
        data_to_write = json.dumps(data_to_write)

        s3.put_object(
            Bucket=bucket, Key=output_path, Body=data_to_write)
        return "wrote final output to s3.\n"
    except Exception as e:
        print(f"error writing to s3: {e}")
        sys.exit(1)


def convert_images_to_pdf(image_paths, pdf_path):
    # Create a new PDF file
    c = canvas.Canvas(pdf_path, pagesize=letter)

    for idx, image_path in enumerate(image_paths, start=1):
        # Open the image file
        img = Image.open(image_path)

        # Calculate the aspect ratio to maintain image proportions
        width, height = letter
        aspect_ratio = width / height
        img_width, img_height = img.size
        img_aspect_ratio = img_width / img_height

        if img_aspect_ratio > aspect_ratio:
            new_width = width
            new_height = new_width / img_aspect_ratio
        else:
            new_height = height
            new_width = new_height * img_aspect_ratio

        # Draw the image onto the PDF
        c.drawImage(image_path, 0, 0, width=new_width, height=new_height)

        # Add a new page for the next image (except for the last one)
        if idx < len(image_paths):
            c.showPage()

    # Save the PDF
    c.save()


def get_ca_name(ca_text):
    system_prompt = """From the text given, can you please tell me that what is the name of Company of Chartered Accountants? Please only give the name of the Chartered Accountants Company. *Very Important: If not found then return ''
    """
    gpt_response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{ca_text}"}
        ],
        max_tokens=20
    )

    return gpt_response['choices'][0]['message']['content']


def write_pdftos3(file_path):
    global tabels_pdf_s3_key

    region = secret[4]
    bucket = secret[3]
    s3_key = tabels_pdf_s3_key
    try:
        with open(file_path, 'rb') as f:
            s3.upload_fileobj(f, bucket, s3_key)

            # print("wrote final output to s3.\n")
    except Exception as e:
        print(f"error writing to s3: {e}")
        sys.exit(1)


def start_extracting(i, comp_id, report_id, length, standardized):
    global tabels_pdf_s3_key
    global table_check
    text = ""
    counter = 1
    tabel_image_pdf_paths = []
    while counter < 10 and i <= length:
        image_path = os.path.join(os.getcwd(), '..', 'temp', 'files', str(
            comp_id), str(report_id), 'images', f'page_{i}.png')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        extracted_text = pytesseract.image_to_string(image)
        extracted_text = extracted_text.replace("’", "'")
        extracted_text = extracted_text.replace("‘", "'")
        extracted_text = extracted_text.replace("”", "")
        extracted_text = extracted_text.replace("“", "")
        original_text = extracted_text
        extracted_text = extracted_text.lower()
        # if i>=6:
        #     print(extracted_text)

        if counter > 1:
            standardized_phrase = "to the independent auditors' report"
            variations = ["""to the auditors' report""", "to auditors' report", "to auditors report", "to the auditors report", "to the independent auditors report",
                          "to the independent auditor's report", "to independent auditor's report", "to the auditor's report", "to independent auditors' report"]
            for variation in variations:
                extracted_text = extracted_text.replace(
                    variation, standardized_phrase)
            if standardized_phrase in extracted_text:
                # print("Breaking- Got auditor's report here")
                break

        if standardized == "annexure ~ b":
            # print("Annexure______B")
            # if "balance sheet" in extracted_text:
            #     print("Balance sheet found")
            #     break
            if "equity and liabilities" in extracted_text:
                # print("Equity and liabilities found")
                break

        if standardized == "annexure ~ a":
            if "|" in extracted_text:
                # print("Tabel-------------->", i)
                tabel_image_pdf_paths.append(image_path)

        text += extracted_text
        if "UDIN" in original_text:
            # print("Found UDIN Breaking")
            i += 1
            break

        i += 1
        counter += 1

    if len(tabel_image_pdf_paths):
        table_check = True

        saved_pdf_path = os.path.join(os.getcwd(), '..', 'temp', 'files', str(
            comp_id), str(report_id), 'images', "table_pdf.pdf")

        convert_images_to_pdf(tabel_image_pdf_paths, saved_pdf_path)
        write_pdftos3(saved_pdf_path)

    return text, i


def s3_downloadfile(report_id, comp_id, file_path):
    file_name = file_path.split("/")[-1]
    home_dir = os.path.expanduser('..')
    # home_dir = os.path.join(os.getcwd(), 'Auditor')
    # print(home_dir)
    # print("Downloading file from S3")
    # Specify your desired directory path here

    destination_directory = os.path.join(
        os.getcwd(), '..', 'temp', 'files',  comp_id, report_id)
    # Make sure the directory exists; create it if it doesn't
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    destination_file_path = os.path.join(destination_directory, file_name)
    # Specify your AWS region
    region = secret[4]
    # Set your AWS credentials directly in the script
    # Upload the file
    # print(file_path, destination_file_path)
    try:
        # print("Downloading file from S3")
        s3.download_file(
            secret[3], file_path, destination_file_path)
        # print(response)
        # print(f"file downloaded successfully {destination_file_path}")
    except ClientError as e:
        # print(f"there was an error on download {file_name} file.")
        report_dir = os.path.join(home_dir, "temp", comp_id, report_id)
        # print("creating empty files")
        with open(os.path.join(report_dir, file_name), 'w') as file:
            file.write(json.dumps({}))

    return destination_file_path


def getting_heading(report_id, comp_id, file_path):
    global final_output

    # Replace with the path to your PDF file
    pdf_file_path = s3_downloadfile(report_id, comp_id, file_path)

    output_folder = os.path.join(
        os.getcwd(), '..', 'temp', 'files', str(comp_id), str(report_id), 'images')
    os.makedirs(output_folder, exist_ok=True)
    # print(pdf_file_path)
    length = pdf_to_images(pdf_file_path, output_folder, dpi=300)
    # print("DONE")
    i = 1
    # Using a single standardized phrase
    standardized_phrases = [
        "independent auditor's report", "annexure ~ a", "annexure ~ b"]
    previours_i = 2
    ca_text = ""
    for standardized_phrase in standardized_phrases:
        # print(i)
        # print(standardized_phrase)
        cnt = 0
        while i < length:
            # print(os.path.join(os.getcwd(), 'images', f'page_{i}.png'))
            image_path = os.path.join(os.getcwd(), '..', 'temp', 'files', str(
                comp_id), str(report_id), 'images', f'page_{i}.png')
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image)

            if i < 3:
                ca_text += extracted_text

            extracted_text = extracted_text.replace("’", "'")
            extracted_text = extracted_text.replace("‘", "'")
            extracted_text = extracted_text.replace("”", "")
            extracted_text = extracted_text.replace("“", "")
            extracted_text = extracted_text.lower()
            if standardized_phrase == "independent auditor's report":
                variations = ["independent auditors' report",
                              "independent auditor's report", "independent auditors report"]
                for variation in variations:
                    extracted_text = extracted_text.replace("*", '')
                    extracted_text = extracted_text.replace(
                        variation, standardized_phrase)

            elif standardized_phrase == "annexure ~ a":
                variations = ["annexure a", "annexure 'a'", "annexure-'a'",
                              "annexure - a", "annexure — a", "annexure-a", "annexure to", "annexure 1"]
                for variation in variations:
                    extracted_text = extracted_text.replace(
                        variation, standardized_phrase)

            elif standardized_phrase == "annexure ~ b":
                variations = ["annexure b", "annexure 'b'", "annexure-'b'",
                              "annexure - b", "annexure — b", "annexure-b", "annexure 2"]
                for variation in variations:
                    extracted_text = extracted_text.replace(
                        variation, standardized_phrase)
                # print("=========================",extracted_text)

            if standardized_phrase in extracted_text:
                # print("I", i)
                text, i = start_extracting(
                    i, comp_id, report_id, length, standardized_phrase)
                if standardized_phrase == "independent auditor's report":
                    previours_i = i
                final_output[standardized_phrase] = text
                # print(text)
                break
                # return final_output
            else:

                if standardized_phrase == "annexure ~ a" or standardized_phrase == "annexure ~ b":
                    cnt += 1
                    if cnt > 5:
                        i = previours_i
                        break

                i += 1
    shutil.rmtree(output_folder)
    # print("=============", pdf_file_path)
    os.remove(pdf_file_path)

    ca_name = get_ca_name(ca_text)

    final_output['ca_name'] = ca_name

    return final_output


def get_comment(data, output_path):
    global table_check
    global gpt_model
    global tabels_pdf_s3_key

    final_output = {}
    try:
        user_input_audit = data["independent auditor's report"]
    except:
        user_input_audit = ""
    try:
        user_input_aneexure_a = data["annexure ~ a"]
    except:
        user_input_aneexure_a = ""
    try:
        user_input_aneexure_b = data["annexure ~ b"]
    except:
        user_input_aneexure_b = ""

    final_output["auditor_name"] = data["ca_name"]

    def check_output(input_text, system_prompt, max_tokens=100):
        # print("Max ", max_tokens)

        # print(system_prompt)
        if input_text == '""':
            # print("Empty")
            return ""
        # time.sleep(30)
        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{input_text}"}
            ],
            max_tokens=max_tokens
        )

        return gpt_response['choices'][0]['message']['content']

    system_prompt_qualified_opinion = """The input is about the Qualified Opinion for the company. Your task is to analyze
    the input thoroughly and if there is any "qualified opinion" mentioned in the paragraph then extract all the text of
    the qualified opinion and change the Language to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them.
    * Important Note:
    Language should be changed to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them
    If not found then return ""
    """
    system_prompt_basis_of_qualified_opinion = """The input is about the basis for qualified opinion for the company. Your task is to analyze
    the input thoroughly and if there is any "basis for qualified opinion" mentioned in the paragraph then extract all the text of the basis for qualified opinion and change the Language to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them.
    * Important Note:
    Language should be changed to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them
    If not found then return ""
    """
    system_prompt_auditor_emphasisofmatters = """From the input provided, Look for
        1. Emphasis of Matter
        If there is any "Emphasis of Matter" mentioned in the paragraph then extract all the text of the emphasis of matter and change the Language to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them’.
        Language should be changed to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them’ and extract all information under emphasis of matter.
        If not found then return ""
        """

    system_prompt_auditor_othermatters = """Donot include the input text in the output.Your task is to look for other matters section and If any negative aspect is highlighted in other matters section then return the information. Negative aspect can be losses, litigations, impact of any fraud, lapse in process controls, etc. If everything is good then return "".
    *Very Important: Do not include the input text in the output.
    * Important: Answer only if information is found otherwise strictly return ""
    """

    system_prompt_auditor_report = """Donot include the input text in the output. If any director disqualification, legal violations, or regulatory fines is highlighted in Report on other Legal and Regulatory Requirements section then return the information. If no director is disqualified, no legal violations, or no regulatory fines then strictly return "".
    *Very Important: Do not include the input text in the output.
    *Very Important: Answer only if information is found otherwise strictly return ""
    """

    system_prompt_anex_B = """ Donot include the input text in the output. The input talks about the opinion to the company. Your task is to look only for opinion section and If Internal financial controls are not operating effectively then return the information. If everything is good then return "".
    *Very Important: Do not include the input text in the output.
    *Very Important: If no information found then return "Data not available"
    """

    if user_input_audit:
        # First checking for qualified opinion
        if "qualified opinion" in user_input_audit:
            # time.sleep(30)
            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{system_prompt_qualified_opinion}"},
                    {"role": "user", "content": f"{user_input_audit}"}
                ],
                max_tokens=1000
            )

            final_output['Independent_Auditor_Report'] = {
                "Qualified_Opinion": gpt_response['choices'][0]['message']['content']}

            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system",
                        "content": f"{system_prompt_basis_of_qualified_opinion}"},
                    {"role": "user", "content": f"{user_input_audit}"}
                ],
                max_tokens=1000
            )

            try:
                final_output['Independent_Auditor_Report']["Basis_for_Qualified_Opinion"] = gpt_response['choices'][0]['message']['content']
            except:
                final_output['Independent_Auditor_Report'] = {
                    "Basis_for_Qualified_Opinion": gpt_response['choices'][0]['message']['content']}

        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{system_prompt_auditor_emphasisofmatters}"},
                {"role": "user", "content": f"{user_input_audit}"}
            ],
            max_tokens=300
        )
        # print(gpt_response['choices'][0]['message']['content'])
        try:
            final_output['Independent_Auditor_Report']["Emphasis_of_matter"] = check_output(
                gpt_response['choices'][0]['message']['content'].replace("json", ""), system_prompt_auditor_emphasisofmatters)
        except:
            final_output['Independent_Auditor_Report'] = {
                "Emphasis_of_matter": check_output(gpt_response['choices'][0]['message']['content'].replace("json", ""), system_prompt_auditor_emphasisofmatters)}

        # time.sleep(30)
        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{system_prompt_auditor_report}"},
                {"role": "user", "content": f"{user_input_audit}"}
            ],
            max_tokens=100
        )
        # print(gpt_response['choices'][0]['message']['content'])
        child_output = check_output(gpt_response['choices'][0]['message']['content'].replace(
            'json', ''), system_prompt_auditor_report)
        if child_output == "" or child_output == "Data not available":
            child_output = ''

        try:
            final_output['Independent_Auditor_Report']["Legal_Regulatory_Requirements"] = child_output
        except:
            final_output['Independent_Auditor_Report'] = {
                "Legal_Regulatory_Requirements": child_output}

    else:
        final_output['Independent_Auditor_Report'] = {
            "Opinion": "",
        }

    if user_input_aneexure_b:
        # time.sleep(30)
        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{system_prompt_anex_B}"},
                {"role": "user", "content": f"{user_input_aneexure_b}"}
            ],
            max_tokens=100
        )
        # print(gpt_response['choices'][0]['message']['content'])
        child_output = check_output(gpt_response['choices'][0]['message']['content'].replace(
            'json', ''), system_prompt_anex_B)
        if child_output == "" or child_output == "Data not available":
            child_output = ''
        final_output['Annexure_B'] = child_output
        # time.sleep(30)
    else:
        final_output['Annexure_B'] = ""
    # ====================================Annexure A==================================================
    systemp_annexure = """You are tasked with extracting specific fields from a document. The document is an annexure to the independent auditors' report and contains various sections detailing different aspects of the company's financial status and compliance. Your task is to extract the text of following fields from the document. 
    Return the output in below format only, while interpreting the below format, treat anything between << and >> as sub instructions to be used to generate it:
    {
        "property": <<property text>>,
        "inventory": <<inventory text>>,
        "investments": <<investments text>>,
        "deposits": <<deposits text>>,
        "frauds": <<frauds text>>,
        "statutory": <<Statuary text>>,
        "party_transactions": <<transactions text>>,
        "internal_audit": <<internal audit text>>,
        "ncash_transactions": <<ncash transactions text>>,
        "rbi_act": <<RBI act text>>,
        "cash_losses": <<cash losses text>>,
        "statuary": <<statuary auditor text>>,
        "transfer_of_funds": <<transfer of funds text>>
    }
    Ensure that the JSON output is well-organized and includes all the necessary text extracted from the document.
    """

    sys_anex_a_statutory_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the statutory dues and taxes paid by the company and as per instructions given within ### tags.
    ###
    1) If Company has delayed/defaulted in payment of statutory dues which are undisputed for more than 6 months
    2) If Company has outstanding statutory dues due to any dispute for more than 6 months
    ###
    If any instruction matched in statutory dues and taxes paid by the company then return 1
    If no instructions are matched in statutory dues and taxes paid by the company then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "statutory": <<output. If no instructions matched then give 0>> 
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_investments_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the Investments made by the company and as per instructions given within ### tags.
    ###
    1. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. without fixed repayment schedule
    2. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has defaulted on loan
    3. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has an outstanding amount of loan due for more than 6 months
    4. If Company has defaulted on loans itself
    ###
    If any instruction matched in investments made by the company then return 1
    If no instructions are matched in investments made by the company then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "investments": <<output. If no instructions matched then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_property_new = """
    The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the Property Plant & Equipment and Intangible Assets of the company and return 1 only if the instructions given within ### tags matches the Property Plant & Equipment and Intangible Assets of the company .
    ###
    1) If the compnay has not maintained documentations of fixed assets.
    2) If the Physical verification of fixed assets is not done by Company.
    3) If physical verification of fixed assets is done by Company but not at appropriate/regular intervals.
    4) If documents of immovable assets are not registered in the name of the Company.
    5) Breach of Benami Transactions (Prohibition) Act, 1988.
    ###
    If any instruction matched in Property Plant & Equipment and Intangible Assets of the company then return 1 
    If no instructions are matched in Property Plant & Equipment and Intangible Assets of the company then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "property": <<output. If no instructions matched then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_inventory_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the inventory of the company and return 1 only if the instructions given within ### tags matches the inventory of the company .
    ###
    1) If Physical verification of inventory is not done by Company.
    2) If Physical verification of inventory is done by Company but not at appropriate/regular intervals.
    3) If material (more than 10%) discrepancy is found within filings/returns filed with banks and books of accounts regarding working capital parameters (Debtors, Creditors, Inventory etc.).
    ###
    If any instruction matched in inventory of the company then return 1
    If no instructions are matched in inventory of the company then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "inventory": <<output. If no instructions matched then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_deposits_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the deposits made by the company and return 1 only if the instructions given within ### tags matches the deposits made by the company .
    ###
    1) If Company has provided loans to subsidiaries/related parties/directors/holding company etc. without fixed repayment schedule.
    2) If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has defaulted on loan.
    3) If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has an outstanding amount of loan due for more than 6 months.
    ###
    If any instruction matched in deposits made by the company then return 1
    If no instructions are matched in deposits then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "deposits": <<output. If no instructions matched then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_frauds_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the frauds made by the company and return 1 only if the instructions given within ### tags matches the frauds made by the company .
    ###
    1) If Company has engaged in or is associated with any company which has conducted a fraud or is suspected of fraud and is under enquiry by government authorities.
    ###
    If any instruction matched in frauds made by the company then return 1
    If no instructions are matched in frauds then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "frauds": <<output. If no instructions matched then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_party_transactions_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the Party Transactions of the company and return 1 only as per instructions given within ### tags.
    ###
    1) If Company's transactions 'fails' under section 188 of the Companies Act, 2013 with respect to maintaining transactions at an arm length basis then only return 1.
    2) If Company's transactions 'fails' under section 177 of the Companies Act, 2013 with respect to maintaining transactions at an arm length basis then only return 1.
    3) If disclosures of related party transactions are *NOT made properly or have been 'misrepresented' or there is an error in the disclosures then only return 1.
    ###
    If any instruction matched for the party transactions of the company then return the data as it is of party transcations.
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "party_transactions": <<output. If no instructions matched for party transcation then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_internal_Audit_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the Internal Audit of the company and return 1 as per instructions given within ### tags. 
    ###
    1) If Company has internal audit mechanism.
    2) If there is a gap within the internal audit mechanism which Company undergoes regularly.
    ###
    If any instruction matched in internal audits of the company then return 1
    If no instructions are matched in internal audits of the company then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "internal_audit": <<output. If no instructions matched for internal audits then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """
    sys_anex_a_ncash_transactions_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the Non cash transcations made by the company and return 1 as per instructions given within ### tags.
    ###
    1) If Company has engaged in non-cash transactions with directors or persons connected with directors.
    ###
    If any instruction matched in Non cash transcations made by the company then return 1
    If no instructions are matched in Non cash transcations made by the company then return 0
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "ncash_transactions": <<output. If no instructions matched for non cash transactions then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    sys_anex_a_RBI_act_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only that If Company is required to be registered under Section 45-IA of the RBI Act, 1934 then return 1 else 0.
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "rbi_act": <<output. If no instructions matched for rbi act then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """
    sys_anex_a_cash_losses_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only that If Company faces cash losses as on the Balance Sheet date then return 1 else return 0.
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "cash_losses": <<output. If no instructions matched for cash losses then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """
    sys_anex_a_statuary_auditor_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only that If there is a resignation of the earlier appointed statutory auditor for any reason then return 1 else 0.
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "statuary": <<output. If no instructions matched for statuary auditor then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """
    sys_anex_a_transfer_of_funds_new = """The text is about the auditors report of a compnay which has several points in it. Your task it to look only for the funds transfered by the company and return 1 as per instructions given within ### tags.
    ###
    1) If for any reason the Company has not spent the requisite minimum amount of CSR expenses required to be spent within the financial year.
    2) If for any reason the Company is not able execute the planned CSR project outlined for the financial year.
    Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
    { 
    "transfer_of_funds": <<output. If no instructions matched for transfer of funds then give 0>>
    }
    <<do not copy content from input to populate the output json but use your own interpretations. do not output anything else but the json.>>
    """

    # =================================================================Old Annexure A prompts===========================================================

    sys_anex_a_statutory = """The input talks about the statutory dues and taxes paid by the company. Your task is to look for statuary dues and taxes point and understand it input clearely and if the conditions below are met then extract the details of the dues and taxes paid by the company.
    The conditions are:
    1. If Company has delayed/defaulted in payment of statutory dues which are undisputed for more than 6 months
    2. If Company has outstanding statutory dues due to any dispute for more than 6 months
    If the above conditinos are met then extract the details of the dues and taxes paid by the company.
    * Important: If any table found then only return the data in table format else return in string format.
    * Important: Also give summary of the content. The summary must be to the point.
    * Important: Give the output in the following json format.
    {
    'table':[{
        'columns': column_names,
        'data': values
    }],
    'writeup': summary
    }
    If the conditions are not met then return "".
    """

    sys_anex_a_investments = """The input talks about the Investments by the company. Your task is to look for inverstement point understand the input clearely and if the conditions below are met then extract the details of Investments by the company.
    The conditions are:
    1. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. without fixed repayment schedule
    2. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has defaulted on loan
    3. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has an outstanding amount of loan due for more than 6 months
    4. If Company has defaulted on loans itself
    * Important: If any table found then only return the data in table format else return in string format.
    * Important: Also give summary of the content. The summary must be to the point.
    * Important: Give the output in the following json format.
    {
    'table':[{
        'columns': column_names,
        'data': values
    }],
    'writeup': summary
    }
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_property = """The input talks about the Property Plant & Equipment and Intangible Assets by the company. Your task is to look for Property Plant & Equipment and Intangible point and understand the input clearely and if the conditions below are met then extract the details of Property Plant & Equipment and Intangible Assets by the company.
        The conditions are:
        1. Improper documentation/records of fixed assets.
        2. If Physical verification of fixed assets is not done by Company.
        3. If physical verification of fixed assets is done by Company but not at appropriate/regular intervals.
        4. If documents of immovable assets are not registered in the name of the Company.
        5. Breach of Benami Transactions (Prohibition) Act, 1988.
        * Important: If any table found then only return the data in table format else return in string format.
        * Important: Also give summary of the content. The summary must be to the point.
        * Important: Give the output in the following json format.
        {
        'table':[{
            'columns': column_names,
            'data': values
        }],
        'writeup': summary
        }
        * Important: If the conditions are not met then return ""
        * Important: Do not return input as output.
        * Important: If not found then return ""
        * Important: If confidence is low then return ""
        """

    sys_anex_a_inventory = """The input talks about the Inventory of the company. Your task is to  look for Inventory point and understand the input clearely and if the conditions below are met then extract the details of Inventory of the company.
    The conditions are:
    1. If Physical verification of inventory is not done by Company.
    2. If Physical verification of inventory is done by Company but not at appropriate/regular intervals.
    3. If material (more than 10%) discrepancy is found within filings/returns filed with banks and books of accounts regarding working capital parameters (Debtors, Creditors, Inventory etc.).
    * Important: If any table found then only return the data in table format else return in string format.
    * Important: Also give summary of the content. The summary must be to the point.
    * Important: Give the output in the following json format.
    {
    'table':[{
        'columns': column_names,
        'data': values
    }],
    'writeup': summary
    }
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """

    sys_anex_a_deposits = """The input talks about the Deposits/Loans/Advances/Guarantees of the company. Your task is to look for Deposits/Loans/Advances/Guarantees point and understand the input clearely and if the conditions below are met then extract the details of Deposits/Loans/Advances/Guarantees of the company.
    The conditions are:
    1. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. without fixed repayment schedule.
    2. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has defaulted on loan.
    3. If Company has provided loans to subsidiaries/related parties/directors/holding company etc. and the borrower has an outstanding amount of loan due for more than 6 months.
    * Important: If any table found then only return the data in table format else return in string format.
    * Important: Also give summary of the content. The summary must be to the point.
    * Important: Give the output in the following json format.
    {
    'table':[{
        'columns': column_names,
        'data': values
    }],
    'writeup': summary
    }
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """

    sys_anex_a_frauds = """The input talks about the Frauds of the company. Your task is to look for Frauds point and understand the input clearely and if the conditions below are met then extract the details of Frauds of the company.
    The conditions are:
    1. If Company has engaged in or is associated with any company (subsidiary/holding company/associate concern/related party/director/shareholder) which has conducted a fraud or is suspected of fraud and is under enquiry by government authorities.
    * Important: If any of the above point satisfies then return the summary of it in a json format.
    * Important: If any table found then return the data in table format else return in string format.
    * Important: Give the output in the following json format.
    {
    'table':[{
        'columns': column_names,
        'data': values
    }],
    'writeup': summary
    }
    If the conditions are not met then return "".
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_party_transactions = """The input talks about the Related Party Transactions of the company. Your task is to look for Party Transactions point and understand the input clearely and if the conditions below are met then extract the details of Related Party Transactions of the company.
    The conditions are:
    1. If Company's transactions breach the provisions under section 177 and 188 of the Companies Act, 2013 with respect to maintaining transactions at an arm length basis.
    2. If disclosures of related party transactions have been misrepresented or there is an error in the disclosures.
    If the conditions are not met then return "".
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_internal_Audit = """The input talks about the Internal Audit of the company. Your task is to look for Internal Audit point and understand the input clearely and if the conditions below are met then extract the details of Internal Audit of the company.
    The conditions are:
    1. If Company's internal audit mechanism does not conform with the size and nature of business
    2. If there is a gap within the internal audit mechanism which Company undergoes regularly
    If the conditions are not met then return "".
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_ncash_transactions = """The input talks about the Non-cash transactions of the company. Your task is to look for Non-cash transactions point and understand the input clearely and if the conditions below are met then extract the details of Non-cash transactions of the company.
    The conditions are:
    1. If Company has engaged in non-cash transactions with directors or persons connected with directors.
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_RBI_act = """The input talks about the Register under RBI Act, 1934 of the company. Your task is to look for Register under RBI Act, 1934 point and understand the input clearely and if the conditions below are met then extract the details of Register under RBI Act, 1934 of the company.
    The conditions are:
    1. If Company is required to be registered under Section 45-IA of the RBI Act, 1934.
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_cash_losses = """The input talks about the Cash Losses of the company. Your task is to look for Cash Losses point and understand the input clearely and if the conditions below are met then extract the details of Cash Losses of the company.
    The conditions are:
    1. If Company incurrs cash losses as on the Balance Sheet date.
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_statuary_auditor = """The input talks about the Resignation of Statutory auditors of the company. Your task is to look for Resignation of Statutory auditors point and understand the input clearely and if the conditions below are met then extract the details of Resignation of Statutory auditors of the company.
    The conditions are:
    1. If there is a resignation of the earlier appointed statutory auditor for any reason.
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """
    sys_anex_a_transfer_of_funds = """The input talks about the Transfer of funds of the company. Your task is to look for Transfer of funds point and understand the input clearely and if the conditions below are met then extract the details of Transfer of funds of the company.
    The conditions are:
    1. If for any reason the Company has not spent the requisite minimum amount of CSR expenses required to be spent within the financial year.
    2. If for any reason the Company is not able execute the planned CSR project outlined for the financial year.
    * Important: If the conditions are not met then return ""
    * Important: Do not return input as output.
    * Important: If not found then return ""
    * Important: If confidence is low then return ""
    """

    # get the data types of column

    def try_convert(value):
        date_formats = ["%Y-%m-%d", "%Y-%m", "%Y", "%Y-%y"]

        if not value.strip():
            return "Empty"

        for date_format in date_formats:
            try:
                datetime.strptime(value, date_format)
                datetime.strptime(value, date_format).date()
                return 'Date'
            except ValueError as e:
                pass
        try:
            int(value)
            return "Number"
        except ValueError:
            pass

        try:
            float(value)
            return "Float"
        except ValueError:
            pass

        pattern = re.compile(r'\b(?:\(\d+,\d+\)|\d+,\d+)\b')
        matches = pattern.findall(value)
        if (matches):
            return 'Number'

        if (value.replace(" ", "") == "-"):
            return "Number"

        if (value.replace(" ", "") == "Nil"):
            return "Number"

        return "String"

    # get the data types of column

    def get_types(array):
        converted_array = [try_convert(item) for item in array]
        return [item for item in converted_array]

    def process_tabels(outpt):

        # print("Output: ", outpt)
        tables = outpt['dataframes']
        summary_lst = outpt['summary_lst']

        # print("tables: ", tables)

        # print("Summaries: ", summary_lst)

        data = [{
            'name':  '' if df.name is None else df.name,
            'columns': df.columns.tolist(),
            'data': df.values.tolist()
        } for df in tables]

        # print("Data : ", data)

        lenData = len(data)
        to_be_removed = []

        # loop for merge tables
        for i in range(lenData):
            try:
                if i < lenData - 1:
                    current_data = data[i]
                    next_data = data[i + 1]
                    last_row = current_data["data"][-1]
                    next_column = next_data["columns"]

                    if len(last_row) == len(next_column):
                        types_array1 = get_types(last_row)
                        types_array2 = get_types(next_column)
                        types_match = all(
                            item1 == item2
                            for item1, item2 in zip(types_array1, types_array2)
                        )
                        if types_match:
                            current_data["data"].append(next_column)

                            for item_data in next_data["data"]:
                                # print("======= item_data", item_data)
                                current_data["data"].append(item_data)
                            to_be_removed.append(i + 1)
            except Exception:
                pass

        # print("\n\n\nData2", data, "\n\n\n")

        data = [item for i, item in enumerate(data) if i not in to_be_removed]

        return data

    def get_int_answer(x):
        # y = 0
        while len(x):
            y = x[-1]
            if y == "0" or y == 0 or y == "1" or y == "1":
                break
            x = x[:-1]

        y = int(y)

        return y

    def get_correct_writeup(writeup, table_there):

        sys_writeup_table = """The following text is the description of beginning of a table of a auditors report of a company. Please act as a financial analyst and make the text in easy and readable language in terms of finance and change the Language to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them’. Make sure that the length of output is always more than input and Language should be changed to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them’.
        """
        if table_there:

            sys_writeup = sys_writeup_table
        else:
            sys_writeup = """The following text is the description of a auditors report of a company. Please act as a financial analyst and make the text in easy and readable language in terms of financeand change the Language to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them’. Make sure that the length of output is always more than input and Language should be changed to third person. For e.g.: ‘We’, ‘us’, ‘our’ will become ‘the auditor’ or ‘their’ or ‘them’."""

        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": f"{sys_writeup}"},
                {"role": "user", "content": f"{writeup}"}
            ],
            max_tokens=200
        )

        return gpt_response['choices'][0]['message']['content']

    def get_table_classification(list_of_classification, data):
        system_get_table_classification = f"""The text is the statement for a particular topic of a auditor's report including tables. Your task is to classify that the text is of which topic out of the {len(list_of_classification)} topics given inside ### tags. Read the table and classify the text into one of the following topics:
        <<do not copy content from input to populate the output. do not output anything else but the classification name only.>>
        ###
        """
        system_get_table_classification += "\n".join(
            list_of_classification)+"\n###"

        # print(system_get_table_classification)
        gpt_response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system",
                 "content": f"{system_get_table_classification}"},
                {"role": "user",
                 "content": f"{data}"}
            ],
            max_tokens=10
        )

        return gpt_response['choices'][0]['message']['content']

    if user_input_aneexure_a:
        try:
            gpt_response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": f"{systemp_annexure}"},
                    {"role": "user", "content": f"{user_input_aneexure_a}"}
                ],
                max_tokens=4000
            )

            annexure_a_data = json.loads(
                gpt_response['choices'][0]['message']['content'].replace('json', ''))

            annexure_a_tabel = []

            try:
                """Reading tabels"""
                if table_check:
                    # This will gives us the tabels
                    raw_tabels = get_pdf_tables(tabels_pdf_s3_key, "auditors")

                    annexure_a_tabel = process_tabels(
                        raw_tabels)  # This will process the tabels

            except Exception as e:
                # raise e
                annexure_a_tabel = []
                # print("Error while reading the tabels from aws textract")
                pass

            final_output['Annexure_A'] = {}
            tabel_list = []
            table_classification = []
            table_keys = ["property", "inventory",
                          "investments", "deposits", "statutory", "frauds"]
            for i in range(len(annexure_a_tabel)):
                try:
                    op = get_table_classification(
                        table_keys, annexure_a_tabel[i])
                except Exception as e:
                    """means table keys got empty [], meaning as all the keys has table"""
                    # raise e
                    op = ""
                    # pass

                # print("------------------>",op)

                if ")" in op:
                    op = op.split(")")[-1].strip().lower()
                else:
                    op = op.strip().lower()

                if op in table_keys:
                    table_keys.pop(table_keys.index(op))

                # print("Table keys: ", table_keys)

                table_classification.append(op)

                if annexure_a_tabel[i]["name"]:
                    sub_table_json = {
                        op: {
                            "table": [
                                {
                                    "columns": annexure_a_tabel[i]['columns'],
                                    "data": annexure_a_tabel[i]['data']
                                }
                            ],
                            "writeup": annexure_a_data[op]+"\n"+annexure_a_tabel[i]['name']
                        }
                    }

                    tabel_list.append(sub_table_json)

            non_table_keys = ["party_transactions", "internal_audit", "ncash_transactions",
                              "rbi_act", "cash_losses", "statuary", "transfer_of_funds"]
            sys_anex_a = {
                "property": sys_anex_a_property_new,
                "inventory": sys_anex_a_inventory_new,
                "investments": sys_anex_a_investments_new,
                "deposits": sys_anex_a_deposits_new,
                "statutory": sys_anex_a_statutory_new,
                "frauds": sys_anex_a_frauds_new,
                "party_transactions": sys_anex_a_party_transactions_new,
                "internal_audit": sys_anex_a_internal_Audit_new,
                "ncash_transactions": sys_anex_a_ncash_transactions_new,
                "rbi_act": sys_anex_a_RBI_act_new,
                "cash_losses": sys_anex_a_cash_losses_new,
                "statuary": sys_anex_a_statuary_auditor_new,
                "transfer_of_funds": sys_anex_a_transfer_of_funds_new
            }

            try:

                adverse_keys = []
                # """Finding adverse keys from the annexure a"""
                try:
                    for key, value in sys_anex_a.items():
                        gpt_response = openai.ChatCompletion.create(
                            model=gpt_model,
                            messages=[
                                {"role": "system",
                                    "content": f"{sys_anex_a[key]}"},
                                {"role": "user",
                                    "content": f"{annexure_a_data[key]}"}
                            ],
                            max_tokens=10
                        )

                        try:
                            x = json.loads(gpt_response['choices'][0]['message']['content'])[
                                key].replace(".", "")
                            # print("Key ", key, repr(x), len(x))

                        except JSONDecodeError as e:

                            x = gpt_response['choices'][0]['message']['content'].strip(
                            )
                            try:
                                x = get_int_answer(x)
                            except Exception as e:
                                # print("passing for key")
                                pass

                        except AttributeError as e:
                            x = json.loads(
                                gpt_response['choices'][0]['message']['content'])[key]

                        # print(x, type(x))
                        try:
                            x = int(x)
                        except ValueError:
                            pass
                        # print(type(x))

                        if x:
                            # print("------------->", key)
                            adverse_keys.append(key)

                except KeyError:
                    pass

                except Exception as e:
                    pass
                # print("\n\n\n Adverse Keys: ", adverse_keys, "\n\n\n")

                for key in adverse_keys:
                    # print("Checking for key", key)
                    if key in non_table_keys:
                        final_output['Annexure_A'][key] = get_correct_writeup(
                            annexure_a_data[key], table_there=False)
                    else:
                        """this means that the key contain table"""

                        if key in table_classification:
                            pass
                        else:
                            final_output['Annexure_A'][key] = {"table": [],
                                                               "writeup": get_correct_writeup(annexure_a_data[key], table_there=False)}

                for item in tabel_list:
                    for key, value in item.items():
                        writeup = value.get('writeup')

                        value["writeup"] = get_correct_writeup(
                            writeup, table_there=True)

                for tabel in tabel_list:
                    final_output['Annexure_A'].update(tabel)

            except Exception as e:
                """If by chance any key not found then pass that key"""
                # raise e
                pass

        except Exception as e:
            # raise e

            final_output['Annexure_A'] = {}

            sys_anex_a = {
                "property": sys_anex_a_property,
                "inventory": sys_anex_a_inventory,
                "investments": sys_anex_a_investments,
                "deposits": sys_anex_a_deposits,
                "statutory": sys_anex_a_statutory,
                "frauds": sys_anex_a_frauds,
                "party_transactions": sys_anex_a_party_transactions,
                "internal_audit": sys_anex_a_internal_Audit,
                "ncash_transactions": sys_anex_a_ncash_transactions,
                "rbi_act": sys_anex_a_RBI_act,
                "cash_losses": sys_anex_a_cash_losses,
                "statuary": sys_anex_a_statuary_auditor,
                "transfer_of_funds": sys_anex_a_transfer_of_funds
            }

            for key, prompt in sys_anex_a.items():
                max_tokens = 300
                if key == "statutory":
                    max_tokens = 1000
                gpt_response = openai.ChatCompletion.create(
                    model=gpt_model,
                    messages=[
                        {"role": "system", "content": f"{prompt}"},
                        {"role": "user", "content": f"{user_input_aneexure_a}"}
                    ],
                    max_tokens=max_tokens
                )

                if str(gpt_response['choices'][0]['message']['content']).strip() == '""':

                    final_output['Annexure_A'][key] = ""
                else:

                    child_output = check_output(
                        gpt_response['choices'][0]['message']['content'].replace("json", ''), prompt, max_tokens)
                    if child_output == "" or child_output == "Data not available":
                        child_output = ''
                    final_output['Annexure_A'][key] = child_output
                # time.sleep(30)

    else:
        final_output['Annexure_A'] = ""

    """Adding year to the final_auditors json"""

    try:
        final_output['year'] = output_path.split(
            "/")[-1].split("_")[-1].split(".")[0]
    except:
        final_output['year'] = ""

    writetos3(final_output, output_path)

    return 1


function_name = "getting_heading"
if (function_name == "getting_heading"):

    global table_check
    table_check = False

    comp_id = "274600185"
    report_id = "542613164"

    tabels_pdf_s3_key = f'files/{comp_id}/{report_id}/gst_extracted_table_locals.pdf'
    print(tabels_pdf_s3_key)
    report = getting_heading(
        report_id=report_id, comp_id=comp_id, file_path="files/274600185/542613164/xbrl/ADSPL_Audit Report FY 2022-23_Signed.pdf")
    op = get_comment(
        data=report, output_path=f"files/{comp_id}/{report_id}/auditor_2023_local.json")

    print(op)

"""Getting_heading will extract all the auditors text from the pdf and return it as a json"""
"""get_comment will take the auditors text and return the json output of adverse comments found"""