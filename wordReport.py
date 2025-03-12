import os
import openai
import json
import time
import pandas as pd
import traceback
import sys
import boto3
from botocore.exceptions import ClientError
import timeout_decorator


openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")
# Set the option to display all columns. None means unlimited.
pd.set_option('display.max_columns', None)
# Set the option to display all rows. None means unlimited.
pd.set_option('display.max_rows', None)
# Increase the width of each column to prevent truncationpd.set_option('display.max_colwidth', None)
aws_access_key_id =  os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key =  os.getenv("AWS_SECRET_ACCESS_KEY")
s3 = boto3.client("s3" , aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
def json_parser(blob):
    if isinstance(blob, list):
        return blob
    parsed = json.loads(blob)
    if isinstance(parsed, str):
        parsed = json_parser(parsed)
    return parsed
# home_dir=os.path.expanduser('..')
# report_dir= os.path.join(home_dir,"temp", sys.argv[3], sys.argv[2])
# prompt_folder=report_dir
prompt_dict={
    "Brief history and line of business": "Brief history and line of business",
    "Business Operations": "business operations",
    "Scale of Operations":"scale of operations",
    "margins":"margins",
    "Capital Structure":"capital structure",
    "liquidity":"liquidity",
    "Working Capital Cycle":"working capital cycle"
}
final_output={
    "Brief history and line of business":"",
    'Business Operations':"",
    "Scale of Operations":"",
    "margins":"",
    "Capital Structure":"",
    "liquidity":"",
    "Working Capital Cycle":""
}
# read file form s3
def readS3file(file_name):
    keyPath = f'prompts/{file_name}'
    response = s3.get_object(Bucket="snuckworks", Key=keyPath)
    # print("===== response", response)
    json_content = response["Body"].read().decode("utf-8")
    # user_prompt_content = json_parser(json_content)
    return json_content

def getCompSummary(company_id , report_id , location, headoffice):
    timeout = 30
    @timeout_decorator.timeout(timeout)  # Set a 20-second timeout
    def run_child_llm_with_timeout(system_prompt, gpt_model, max_tokens=250):
        global timeout
        # print(f"timeout = {timeout}")
        try:
            gpt_response=openai.ChatCompletion.create(
                    model=gpt_model,
                    messages=[
                        {"role": "system", "content": f"{system_prompt}"}
                    ],
                    max_tokens=max_tokens
            )
            return gpt_response
        except openai.error.RateLimitError as e:
            # print(f"Rate limit error in LLM. {e}")
            if('You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.' in str(e)):
                print(e)
                sys.exit(1)
            
            time.sleep(15)
            pass
        except timeout_decorator.TimeoutError:
            # print("LLM timed out.")
            return {"timeouterror":0}
        except Exception as e:
            # print("Other LLM error.")
            traceback.print_exc()
            return {"timeouterror":0}
    def run_llm(system_prompt, gpt_model, max_tokens=250):
        run_idx = 0
        #rerun_counter = 1
        max_rerun = 3
        while run_idx < max_rerun:
            gpt_response = run_child_llm_with_timeout(system_prompt, gpt_model, max_tokens)
            if 'timeouterror' in gpt_response.keys():
                # print(f"Rerun count {run_idx}")
                pass
            else:
                return gpt_response
            run_idx += 1
        return {"timeouterror":0} #if code came here means not a single llm run was successful
    def run_gpt(system_prompt, max_tokens=250):
        gpt_model = "gpt-3.5-turbo"
        doc_parts = 1
        if gpt_model == "gpt-4":
                time.sleep(int(60/doc_parts) + 2) #to overcome LLM token rate limit
        else:
                time.sleep(1)
        #### analyse the data now
        try:
            gpt_response=run_llm(system_prompt, gpt_model, max_tokens)
            if 'timeouterror' not in gpt_response.keys():
                tokens = gpt_response['usage']['total_tokens']
                # print(f"-----tokens used: {tokens}-----")
                # print(f"-----every answer: {gpt_response['choices'][0]['message']['content']}")
                if gpt_model != 'gpt-4':
                        dt=gpt_response['choices'][0]['message']['content']
                        fixed_json_str = dt.replace("\'", "\"")
                        first_level_reponse=fixed_json_str
                else:
                        dt=gpt_response['choices'][0]['message']['content']
                        fixed_json_str = dt.replace("\'", "")
                        first_level_reponse=fixed_json_str
                #print(f"-----every answer: {first_level_reponse}")
                return first_level_reponse
        except openai.error.RateLimitError as e:
                # print(f"Rate limit error in LLM. {e}")
                if('You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.' in str(e)):
                    print(e)
                    sys.exit(1)
                time.sleep(15)
                pass
        except openai.error.InvalidRequestError as e:
                # print(f"LLM error. {e}")
                traceback.print_exc()
                return ""
        except openai.error.ServiceUnavailableError as e:
                # print(f"LLM error. {e}")
                traceback.print_exc()
                return ""
        except Exception as e:
                # print(f"Token limit error in LLM. {e}")
                traceback.print_exc()
                return ""
    # Initialize the final_output dictionary
    
 
    for prompt_key in prompt_dict:
        keyPath = f'files/summary/{company_id}/{report_id}/{prompt_dict[prompt_key]}.json'
        response = s3.get_object(Bucket="snuckworks", Key=keyPath)
        # print("===== response", response)
        json_content = response["Body"].read().decode("utf-8")
        # user_prompt_content = json_parser(json_content)
        user_prompt_content = json_content
        system_prompt_content= readS3file(prompt_dict[prompt_key])
        # Combine user and system prompt content

        print({"\n\n\n", "{prompt_key}",system_prompt_content, "\n\n\n", user_prompt_content, "\n\n\n"})
        full_prompt_content = system_prompt_content + user_prompt_content
        # print("**************USER PROMPT**************")
        # print(f"prompt by user:  {prompt_dict[prompt_key]} \n",user_prompt_content)
        # print("**************SYSTEM PROMPT**************")
        # print(f"prompt by system : {prompt_dict[prompt_key]} \n",system_prompt_content)
        # Call the run_gpt function with the combined prompt
        response = run_gpt(full_prompt_content)
        # Store the response in the final_output dictionary
        # REPLACE " WITH ' IN THE RESPONSE
        response = response.replace('"', "'")
        final_output[prompt_key] = response
    
    if(location):
        system_prompt = """The input is the address. So from the input address given under ### tages, can you please tell me what is the city or district name? Please only give the name of the city or district. 
        *Very Important: If not found then return '' 
        """
        user_prompt = location

        full_prompt_content = system_prompt + "\n".join(user_prompt)+"\n###"
        response = run_gpt(full_prompt_content)
        response = response.replace('"', "'")
        final_output['city_name'] = response
    else:
         final_output['city_name'] = ""
    
    if(headoffice):
        system_prompt = """The input is the address. So from the input address given under ### tages, can you please tell me what is the city or district name? Please only give the name of the city or district. 
        *Very Important: If not found then return '' 
        """
        user_prompt = headoffice

        full_prompt_content = system_prompt+ "\n".join(user_prompt)+"\n###"
        response = run_gpt(full_prompt_content)
        response = response.replace('"', "'")
    else:
         final_output['city_name_'] = ""
    
    return final_output


function_name = sys.argv[1]
if(function_name == "getCompSummary"):
      response = getCompSummary(sys.argv[2] , sys.argv[3], sys.argv[4] , sys.argv[5])
      print(json.dumps(response,indent=4))
# if __name__ == "__main__":
#     print(getCompSummary("1234","123"))