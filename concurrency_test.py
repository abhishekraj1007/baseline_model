from threading import Thread
import requests
import time

payload = {
    "finalQuizData": {
        "first time": {
            "qno": 2,
            "question": "Are you shopping with us for the 1st time?",
            "attribute": True,
            "value": True
        },
        "size": {
            "qno": 3,
            "question": "Could we get your digits?",
            "attribute": {
                "Bust": "34-35",
                "Hips": "36-37.5",
                "Waist": "25-26.5"
            },
            "value": {
                "Bust": "34-35",
                "Hips": "36-37.5",
                "Waist": "25-26.5"
            }
        },
        "Bodies": {
            "qno": 4,
            "question": "How would you describe yours?",
            "attribute": [
                "Apple"
            ],
            "value": [
                "Apple"
            ]
        },
        "accentuate": {
            "qno": 5,
            "question": "Your favourite features that you like to accentuate with your clothing?",
            "attribute": "",
            "value": ""
        },
        "uncomfortable": {
            "qno": 6,
            "question": "Some features that you're not-so-comfortable showcasing in your clothing?",
            "attribute": "",
            "value": ""
        },
        "height": {
            "qno": 7,
            "question": "Would you say you're vertically gifted or efficient?",
            "attribute": "",
            "value": ""
        },
        "colour palettes": {
            "qno": 8,
            "question": "What colour palettes are you most attracted to?",
            "attribute": "",
            "value": ""
        },
        "prints_fan": {
            "qno": 9,
            "question": "Are you a fan of Prints?",
            "attribute": "",
            "value": ""
        },
        "prints": {
            "qno": 10,
            "question": "What kind of prints are you attracted to?",
            "attribute": "",
            "value": ""
        },
        "spend categories": {
            "qno": 11,
            "question": "How much do you want to spend on items from these categories?",
            "attribute": {
                "Accessories": [
                    100,
                    10000
                ],
                "Bottoms": [
                    100,
                    10000
                ],
                "Dresses": [
                    100,
                    10000
                ],
                "Loungewear": [
                    100,
                    10000
                ],
                "Tops": [
                    100,
                    10000
                ]
            },
            "value": {
                "Accessories": [
                    100,
                    10000
                ],
                "Bottoms": [
                    100,
                    10000
                ],
                "Dresses": [
                    100,
                    10000
                ],
                "Loungewear": [
                    100,
                    10000
                ],
                "Tops": [
                    100,
                    10000
                ]
            }
        },
        "styles": {
            "qno": 12,
            "question": "Which one would you describe as your style?",
            "attribute": "",
            "value": ""
        },
        "occasion specific": {
            "qno": 13,
            "question": "Are you shopping for a special ocassion?",
            "attribute": "",
            "value": ""
        },
        "occasion": {
            "qno": 14,
            "question": "Are you shopping for a specific ocassion?",
            "attribute": "Party",
            "value": "Party"
        },
        "email": {
            "value": "chandan.roy@algoscale.com"
        },
        "start easy": {
            "qno": 1,
            "question": "Let's start easy, what brings you here today?",
            "attribute": "Just browsing.",
            "value": "Just browsing."
        },
        "dob": {}
    }
}

NUM_REQUESTS = 500
SLEEP_COUNT = 0.1
REST_API_URL = "http://localhost:5000/personalize"
def call_predict_endpoint(n):
	# load the input image and construct the payload for the request
	# payload = pickle.load(open('payload', 'rb'))
	# submit the request
	r = requests.post(REST_API_URL, json=payload)
	# ensure the request was sucessful
	if r.status_code==200:
		print("[INFO] thread {} OK".format(n))
	# otherwise, the request failed
	else:
		print("[INFO] thread {} FAILED".format(n))
# loop over the number of threads
for i in range(0, NUM_REQUESTS):
	# start a new thread to call the API
	t = Thread(target=call_predict_endpoint, args=(i,))
	t.daemon = True
	t.start()
	time.sleep(SLEEP_COUNT)
# insert a long sleep so we can wait until the server is finished
