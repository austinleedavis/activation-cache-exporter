import http.client
import json
import logging


class Ntfy:

    logger = logging.getLogger(__name__)
    topic: str

    def __init__(self, topic: str = None):
        """Initializes the Ntfy instance with an optional topic.

        If no topic is specified, a warning is logged and notifications will be logged instead of
        sent.
        """
        self.topic = topic
        if topic is None:
            self.logger.warning("No topic specified. Ntfy will use Logger instead.")
            self.send_notification = lambda *a, **k: self.logger.info(str((a, k)))

    def send_notification(self, message, extra_headers={}):
        """Sends a notification with the given message to the specified topic.

        If the topic does not start with a "/", it is prepended. Logs the result of the notification
        attempt.
        """

        if len(self.topic) > 0 and self.topic[0] != "/":
            self.topic = "/" + self.topic

        try:
            # Define the URL and path
            url = "ntfy.sh"

            headers = {"Content-Type": "text/plain"}
            headers.update(extra_headers)

            conn = http.client.HTTPSConnection(url)
            conn.request("POST", self.topic, message.encode("utf-8"), headers)
            response = conn.getresponse()
            if response.status == 200:
                self.logger.info(f"Notification sent successfully: {message}")
            else:
                self.logger.info(
                    f"Failed to send notification. Status code: {response.status}, Reason: {response.reason}, Message: {message}"
                )
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Ensure the connection is closed
            conn.close()

    def subscribe(self, callback):
        """Subscribes to the Ntfy topic and invokes callback when a message is received."""
        if not self.topic:
            return

        if self.topic[0] != "/":
            self.topic = "/" + self.topic

        url = "ntfy.sh"
        conn = http.client.HTTPSConnection(url)

        try:
            conn.request("GET", self.topic + "/json")
            response = conn.getresponse()
            for line in response:
                try:
                    data = json.loads(line)
                    if "message" in data:
                        callback(data["message"])
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            self.logger.error(f"Subscription error: {e}")
        finally:
            conn.close()
