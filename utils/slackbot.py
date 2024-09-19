import configparser
import requests
import json
import os

ABS_PATH = os.path.dirname(__file__)

class Slackbot:
    def __init__(self, bot_name='trainingbot', channel_name='training'):
        self.bot_name = bot_name
        self.channel_name = channel_name

        config = configparser.ConfigParser()
        config.read(f"{ABS_PATH}/args.txt")
        self.token = config['slack']['token']

    def sendMsg(self, text):
        return requests.post(
            'https://slack.com/api/chat.postMessage', {
                'token': self.token,
                'channel': f'#{self.channel_name}',
                'text': text,
                'blocks': None
           }).json()
