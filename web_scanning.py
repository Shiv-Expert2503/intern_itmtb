import sys
import requests
from bs4 import BeautifulSoup
import os
import openai
import json
import time
from utils_web_scrap import saveLogger

from dotenv import load_dotenv

load_dotenv()

no_of_tries=0


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_API_ORG")


headers= {
"Content-Type": "application/json; charset=utf-8",
#"Cookie": "key=value; JSESSIONID=web10102~819MiANTk7yWjC+iDoLKUgJL.2630cd14-e63d-3bc7-b90c-e393b308a14e; SERVERID=ha101",
"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
}

ignore_list=['investor','contact','sitemap','media','download','legal','career','csr','sustainability','privacy','covid','blog']
session = requests.Session()
# Send an HTTP GET request to the URL
def get_html_from_url(url):
    ## function to get html from given url
    response = session.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page using Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    else:
        print("Failed to retrieve the web page.")
        
        sys.exit(1)
        # return 1
    
def get_body_content(soup):
    # Find the <body> tag and extract its contents from html soup
    body_tag = soup.find('body')

    # Check if the <body> tag was found
    if body_tag:
            # Print the contents of the <body> tag
            # print(len(body_tag.text))
            #print(body_tag.text)
            return body_tag
    else:
            print("No <body> tag found on the page.")
            sys.exit(1)
            # return -1


def extract_urls_from_body(body, domain):
    # extract all urls from body tag
    ignore_list=['investor','contact','sitemap','media','download','legal','career','csr','sustainability','privacy','covid','attachment']
    if body:
        anchor_tags = body.find_all('a')
        
        # Extract the 'href' attribute from each <a> tag
        urls = [a.get('href') for a in anchor_tags if a.get('href')]
        urls = list(dict.fromkeys(urls))   
        
        returl=[]
        ##-- keep only urls that are on the company domain, direct or relative
        for url in urls:
            if domain in url:
                returl.append(url)
            elif (url.find('/')==0) and (url !='/' ):
                returl.append(url)

              
        final_ret_url=[]
        for url in returl:
            match=0
            for check_ignore in ignore_list:
                #print(f"comparing {url} to {check_ignore}")                
                if url.lower().find(check_ignore) > -1:
                    match = 1
                    break
                    
            if match==0:
                #print(f"adding {url} to output")
                final_ret_url.append(url)
                   
        return final_ret_url
    else:
        return []
    
page_categories=['Management/Leadership',
'Business',
'Services',
'Product',
'Revenues',
'Location',
'Awards',
'certifications',
'Customers'
]

system_content_url_interpret_summary="""
A list of urls will be provided. You have to guess which category a url belongs to.

The following categories exist:
About Company
Management/Leadership
Business
Services
Product
Revenues
Location
Awards
certifications
Customers

Output all urls in one json with format as follows. instructions are in <<>> tags:
   {
   url 1:<<category this url may belong to. Only guess when confidence is high, else say NA>>,
    url 2:<<category this url may belong to. Only guess when confidence is high, else say NA>>
    }
"""

#------- ORIG ---------
system_content_first_level_summary="""
From the given website's code extract info about the website's businessas per instructions given within ### tags.
###
1) interpreting text in all html tags like div, heading, paragraph etc that are used to show text in a website, and interpret what business this website might be engaged in. Only respond if confidence is high.  
2) if you're not able to guess the business with high confidence from this page's code, look for any texts or labels that may indicate they may contain information about the business. These may be called Products, or Services, or Expertise, or Businesses, or anything similar. Interpret if any of the heading or menu option may sound like resembling the above options. These headings or labels can be found as sections in the website, or as top level options in some menu bar on the website.  If you find any such product sections, list all product categories, associated subcategories, all the way down ot individual products. Similarly list all services or areas of expertise. List out any businesses mentioned.
3) find products, services offered by the company. include any areas of expertise under services.Only respond if your guess confidence is high.
4) find company's customers. if found, give names. if confidence is low, say NA
5) find locations of company. for every location also identify if its a factory(can also be referred to as Plant), office, headquarter, registered office (can also be referred to as Regd Off). if location found, give complete address. if confidence is low, say NA
###
Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
{ 
Business: <<Use your interpretation of this website's business to populate this. summarise company's purpose in upto 100 words. if you are not sure, say NA>>, 
Sector: <<one liner about the sector in which company operates>>,
Activities: <<list all activities website seems to talk about. for ex manufacturing, quality control, R&D, IT development etc>>
Products: <<list all products. If no products are found then give NA>>,
Services: <<list all services. If no services are found then give NA>>,
Customers:<<list all customer names. If not found then give NA>>
Locations:<<list all locations company seems to operate from. If not found then give NA>>
Email:<<any company email if provided. NA if not>>
Phone:<<any company phone if provided. NA if not>>
}
<<do not copy content from website to populate the output json but use your own interpretations. do not output anything else but the json.>>
"""
system_content_first_level_summary="""
From the given website's code extract info about the website's business as per instructions given within ### tags.
###
1) interpreting text in all html tags like div, heading, paragraph etc that are used to show text in a website, and interpret what business this website might be engaged in. Only respond if confidence is high.  
2) if you're not able to guess the business with high confidence from this page's code, look for any texts or labels that may indicate they may contain information about the business. These may be called Products, or Services, or Expertise, or Businesses, or anything similar. Interpret if any of the heading or menu option may sound like resembling the above options. These headings or labels can be found as sections in the website, or as top level options in some menu bar on the website.  If you find any such product sections, list all product categories, associated subcategories, all the way down to individual products. Similarly list all services or areas of expertise. List out any businesses mentioned.
3) find products, services offered by the company. include any areas of expertise under services.Only respond if your guess confidence is high.
4) find company's customers. if found, give names. if confidence is low, say NA
5) find locations of company. for every location also identify if its a factory(can also be referred to as Plant), office, headquarter, registered office (can also be referred to as Regd Off). if location found, give complete address. if confidence is low, say NA
###
Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
{ 
Business: <<Use your interpretation of this website's business to populate this. summarise company's purpose in upto 100 words. if you are not sure, say "Difficult to find from home page.">>, 
Sector: <<one statement about the sector in which company operates>>,
Industry: <<based on detected sector categories it within - (IT, AUTO, PHARMA). If you are not sure aboutt he category then categorise as GENERAL  >>,
Activities: <<list all activities website seems to talk about. for ex manufacturing, quality control, R&D, IT development etc>>,
Products: <<list all products. If no products are found then give NA>>,
Services: <<list all services. If no services are found then give NA>>,
Customers:<<One liner about all customer names. If not found then give "Difficult to find from home page.">>
Locations:<<list all locations company seems to operate from. If not found then give "Difficult to find from home page.">>
Email:<<any company email if provided. "Difficult to find from home page." if not>>
Phone:<<any company phone if provided. "Difficult to find from home page." if not>>
}
<<do not copy content from website to populate the output json but use your own interpretations. do not output anything else but the json.>>
"""
system_content_management_summary="""
From the given website's code extract info about the business' leadership, certifications, awards, locations, incorporation year.
Share your interpretations as a json with following format. while interpreting the json format below, treat anything between << and >> as sub instructions to be used to generate it:
{
    "leadership":<<persons name:summary of this person in under 50 words>>,
    "certifications":<<list of certifications>>,
    "awards":<<list of awards>>,
    "incorporation year":<<statement mentioning year in which company was incorporated>>
    "locations":[
    {
        <<type:factory/plant, office, registered office, headquarters>>,
        <<address:address>>
    },
    ...
]

}

<<Only provide any answer when confidence is high, else skip>>
"""

def get_first_level_summary(parent_url):
    try:
        global no_of_tries
        no_of_tries+=1
        soup=get_html_from_url(parent_url)
        if isinstance(soup, int):
            # print("Error in retrieving data. Check the url and try again")
            exit(-1)

        body_tag=get_body_content(soup)
        url_list=extract_urls_from_body(body_tag, parent_url)

        ## Get first level summary
        if isinstance(body_tag, int):
            # print("Error in retrieving data. Check the url and try again")
            exit(-1)

        user_content=body_tag.text

        if len(user_content) > 15000:
            user_content=user_content[:15000]

        gpt_response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
                {"role": "system", "content": f"{system_content_first_level_summary}"},
                {"role": "user", "content": f"{user_content}"}
            ],
            temperature=1,
                max_tokens=8000,
                top_p=1,
        )

        #print(gpt_response['choices'][0]['message']['content'])
        first_level_reponse=json.loads(gpt_response['choices'][0]['message']['content'])
        first_level_reponse["Website"] = parent_url
        res=json.dumps(first_level_reponse,indent=4)

        #first_level_reponse = json.dumps(first_level_reponse, indent=4)
        #print(f"Tokens used: {gpt_response['usage']['total_tokens']}")
        #print(first_level_reponse)
        
        # return res
        print(res)
        sys.exit(0)
    except Exception as e:
        saveLogger(f"Error occurred in get_first_level_summary: {e}")
        # print(e)
        # print(no_of_tries)
        if "SSLCertVerificationError" in str(e) and no_of_tries<2:
            # print("SSL Certificate error. Try again after some time.")
            saveLogger("SSL Certificate error. Trying with http instead of https.")
            modified_url=parent_url.replace("https:","http:")
            get_first_level_summary(modified_url)

            # print(modified_url)
        
        else:
            saveLogger("Error occurred in get_first_level_summary")
            sys.exit(1)

# print(get_first_level_summary("https://www.cjpl.in/"))

# this website will give max token error , so use 16k 
# https://www.qps.com
    
if __name__=="__main__":
    weburl=sys.argv[1]
    get_first_level_summary(weburl)
