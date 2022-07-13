from threading import Thread
import requests
import time

NUM_REQUESTS = 500
SLEEP_COUNT = 0.1
REST_API_URL = "http://localhost:5000/recommend?email=chandan.roy@algoscale.com&product_title=Carla Mauve Silk Corset Top"
def call_predict_endpoint(n):
	# load the input image and construct the payload for the request
	# payload = pickle.load(open('payload', 'rb'))
	# submit the request
	r = requests.get(REST_API_URL)
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
