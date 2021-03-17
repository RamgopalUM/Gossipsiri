import json
import uuid
from yahoo_stock import *
import boto3
from botocore.config import Config
from datetime import datetime

S3_CONFIG = Config(
    region_name = 'us-east-2',
    retries = {
        'max_attempts': 3,
        'mode': 'standard'
    }
)

def lambda_handler(event, context):
    # fetch action from json
    req = json.loads(event['body'])['queryResult']

    intent = req['intent']['displayName']
    out_text = "hello world"

    if intent == 'Get stock price':
        company = req['parameters']['company_name']
        price = get_realtime_price(company)
        change = get_change(company)
        direction = 'up' if change > 0 else 'down'
        out_text = 'The stock price of ' + symbol_to_name(company) + ' is ' + direction + ' $' + str(abs(change)) + ' today, '
        out_text += 'with a current price of $' + str(round(price, 2)) + '.'
    elif intent == 'Get quote':
        company = req['parameters']['company_name']
        out_text = get_quote(company)
    elif intent == 'Get valuation':
        company = req['parameters']['company_name']
        out_text = get_valuation(company)
    elif intent == 'Get analysis':
        company = req['parameters']['company_name']
        out_text = get_stock_analysis(company)
    elif intent == 'Biggest changers':
        num = int(req['parameters']['number'])
        changer = req['parameters']['changer']
        func_d = {'biggest gainers': get_gainers, 'biggest losers': get_losers, 'most active stocks': get_most_active}
        res = func_d[changer](num)
        out_text = 'Today\'s ' + str(num) + ' ' + changer + ' are:\n'
        for i in range(len(res)):
            symb = '+' if res.iloc[i, 4] > 0 else ''
            out_text += str(i+1) + '. ' + res.Name[i] + ' at ' + symb + str(res.iloc[i, 4]) + '%\n'
    elif intent.split()[-1] == 'plot':
        filename = str(uuid.uuid4())[:10] + '.png'
        s3_client = boto3.client('s3', config=S3_CONFIG)
        location = {'LocationConstraint': 'us-east-2'}
        company_list = req['parameters']['company_name']
        start_date = req['parameters']['start_date']
        end_date = req['parameters']['end_date']
        start = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
        end = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')
        if intent == 'Get multiple plot':
            plot_multiple_comps_price(filename, company_list, start, end)
        elif intent == 'Get return analysis plot':
            return_analysis(filename, company_list, start, end)
        elif intent == 'Get volatility analysis plot':
            volatility_analysis(filename, company_list, start, end)
        s3_client.upload_file('/tmp/' + filename, 'wall-street-digest-media', filename)
        out_text = filename
    elif intent == 'Get sentiment':
        company = req['parameters']['company_name']
        #TODO call backend to get sentiment score
    elif intent == 'Get option':
        company = req['parameters']['company_name']
        date = req['parameters']['date']
        out_text = get_option_table(company, date)
    else:
        out_text = 'Sorry, I didn\'t understand that. Please try asking again'

    body = {
        "fulfillmentText": out_text
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }
    }

    return response
