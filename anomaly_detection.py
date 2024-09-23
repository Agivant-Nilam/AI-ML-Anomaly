from datetime import timedelta
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime, AccumulationMode
from apache_beam.transforms.window import FixedWindows
from datetime import datetime as dt
import vertexai
import json
import jsonlines
from google.cloud import language_v1
import re
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyzeSentiment(beam.DoFn):
    def __init__(self, project, location, model_name):
        self.project = project
        self.location = location
        self.model_name = model_name
        # self.bucket_name = bucket_name

    def setup(self):
        import vertexai
        from vertexai.language_models import TextGenerationModel

        vertexai.init(project=self.project, location=self.location)
        self.model = TextGenerationModel.from_pretrained(self.model_name)
        self.parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }

    def process(self, element):
        # from google.cloud import storage
        try:
        # Check if the element is a dictionary, if so, use it directly
            if isinstance(element, dict):
                record = element
            else:
            # Parse the JSON from the element if it's a string
                record = json.loads(element)

        # Extract the 'EmailText' field
            text = record.get('EmailText', '')
            if text:
                prompt = f"""
                Analyze the sentiment of the provided text.
                Classify the sentiment into one of the following categories: Extremely Negative, Negative, Neutral, Positive, or Extremely Positive.

                Categories:
                Extremely Negative: Indicates a very strong negative sentiment.
                Negative: Indicates a negative sentiment but not as intense as 'Extremely Negative'.
                Neutral: Indicates a neutral sentiment, where the text neither expresses a positive nor negative sentiment.
                Positive: Indicates a positive sentiment but not as intense as 'Extremely Positive'.
                Extremely Positive: Indicates a very strong positive sentiment.

                input: \"Hi Support, I\'m dissatisfied with the quality of product X in order #01758 #Unknown. I am extremely disappointed with this issue.
                This is completely unacceptable. .
                Please address this issue immediately.\"

                Classify the sentiment of the message: Extremely Negative

                input: \"Hi Customer Support, I received my order #12345 but the product quality is below expectations #12345. I am quite unhappy with this issue.
                This is very concerning. It arrived damaged.
                Please address this issue promptly.\"

                Classify the sentiment of the message: Negative

                input: Hi, I\'m concerned about website security while accessing order #04784


                Classify the sentiment of the message: Neutral

                input: \"Hi, Can I request gift wrapping for order #00762? #Unknown. I am very pleased with this.
                This is wonderful news. .
                Please keep up the great work.\"

                Classify the sentiment of the message: Positive

                input: \"Hello, Could you provide detailed specifications for product X in order #03701? #Unknown. I am very pleased with this.
                This is wonderful news. .
                Please keep up the great work.\"

                Classify the sentiment of the message: Extremely Positive

                input: \"Hi Support, I\'m dissatisfied with the quality of product X in order #01758 #Unknown. I am extremely disappointed with this issue.
                This is completely unacceptable. .
                Please address this issue immediately.\"

                Classify the sentiment of the message: Extremely Negative

                input: Hello Support, I\'d like to request a return for my recent order. (Order #02749).

                Classify the sentiment of the message: Neutral

                input: \"Hello, I\'m writing to inquire about my order\'s shipping status #00849wn. I am very pleased with this.
                This is wonderful news. (Order #01173).
                Please keep up the great work.\"

                Classify the sentiment of the message: Extremely Positive

                input: \"Hello Support, Can you apply a discount to order #00214? #Unknown. I am very pleased with this.
                This is wonderful news. .
                Please keep up the great work.\"

                Classify the sentiment of the message: Positive

                input: {text}

                Classify the sentiment of the message:
                """
                response = self.model.predict(prompt, **self.parameters)
                sentiment = response.text.strip()
                # update record with the sentiment and yield the record
                record['Sentiment'] = sentiment
            else:
                logger.warning("No 'EmailText' found in the element")
                record['Sentiment'] = 'Neutral'
        # yield data
            yield record
        except Exception as e:
            logger.error(f"Error in AnalyzeSentiment: {e}")
            record['Sentiment'] = 'Neutral'  # Default value
            yield record
    

class AnalyzeClassification(beam.DoFn):
    def __init__(self, project, location, model_name):
        self.project = project
        self.location = location
        self.model_name = model_name
        # self.bucket_name = bucket_name

    def setup(self):
        import vertexai
        from vertexai.language_models import TextGenerationModel

        vertexai.init(project=self.project, location=self.location)
        self.model = TextGenerationModel.from_pretrained(self.model_name)
        self.parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0,
            "top_p": 1,
            "top_k": 9
        }

    def process(self, element):
        import json

        # for record in element:
        # text = record.get('EmailBody', '')
        try:
            text = element.get('EmailText', '')
            if text:
                prompt = f"""
                Multi-choice problem: Define the category of the ticket?
                Categories:
                Communication Gap

                Complicated Return Processes
                Customer Support
                Damaged Goods
                Damaged Products
                Defective Items
                Delayed Deliveries
                Delayed Refunds
                False Advertising
                Inconsistent Product Descriptions
                Incorrect Charges
                Incorrect Orders
                Incorrect Product Details
                Late Arrivals
                Lost Shipments
                Misleading Information
                Payment Processing Issues
                Poor Customer Service
                Poor Quality Products
                Product Inquiry
                Product Mismatch
                Product Not as Described
                Return and Refund Management
                Rude or Unhelpful Representatives
                Shipping Inquiry
                Slow Response Times
                System Errors
                Technical Issues
                Unresolved Issues
                Website Problems

                Please only print the category name without anything else.

                Ticket: \"Dear Customer Support,

                I recently received my order (Order ID: #90680) and I am pleased to report that the shipping process was smooth. The delivery was timely and the package arrived in excellent condition. Thank you for the great service!\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I am extremely impressed with the shipping service for my order (Order ID: #72836). The delivery was exceptionally fast and the package arrived in pristine condition. Kudos to your team for the outstanding service!\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the shipping process for my order (Order ID: #18543). The package is lost and the delivery was a nightmare. This is unacceptable and needs to be addressed immediately.\"

                Category: Lost Shipments


                Ticket: \"Dear Customer Support,

                I\'m writing to express my frustration with the shipping process for my order (Order ID: #31527). The delivery was delayed and the package arrived damaged. This experience has been quite disappointing.\"

                Category: Damaged Goods


                Ticket: \"Dear Team,

                I wanted to let you know that my order (Order ID: #15773) arrived on time and in perfect condition. I\'m very happy with the shipping service provided. Thank you!\"

                Category: Customer Support


                Ticket: \"Hello,

                I am writing to express my extreme dissatisfaction with the shipping of my order (Order ID: #92034). The package was severely delayed and arrived in a damaged state. This is very disappointing and requires immediate attention.\"

                Category: Damaged Goods


                Ticket: \"Dear Team,

                The shipping process for my order (Order ID: #37494) was unsatisfactory. There were delays and the package was not handled properly. I\'m not happy with the service provided.\"

                Category: Delayed Deliveries


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the shipping of my order (Order ID: #69815). The delivery was lightning fast and the item was perfectly protected. This is by far the best shipping service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the shipping of my order (Order ID: #35886). The delivery was lightning fast and the item was perfectly protected. This is by far the best shipping service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Dear Team,

                My experience with the shipping of order (Order ID: #94668) has been terrible. The package was late, damaged, and the entire process was a hassle. I demand immediate resolution.\"

                Category: Damaged Goods


                Ticket: \"Dear Team,

                The shipping process for my order (Order ID: #10590) was unsatisfactory. There were delays and the package was not handled properly. I\'m not happy with the service provided.\"

                Category: Delayed Deliveries


                Ticket: \"Hello,

                I\'m writing to express my satisfaction with the shipping process for my recent order (Order ID: #93400). The delivery was prompt and the item was well-packaged. Great job!\"

                Category: Customer Support


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the shipping of my order (Order ID: #71285). The delivery was lightning fast and the item was perfectly protected. This is by far the best shipping service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Hello,

                I\'m writing to express my satisfaction with the shipping process for my recent order (Order ID: #79714). The delivery was prompt and the item was well-packaged. Great job!\"

                Category: Customer Support


                Ticket: \"Dear Team,

                The shipping process for my order (Order ID: #74804) was unsatisfactory. There were delays and the package was not handled properly. I\'m not happy with the service provided.\"

                Category: Delayed Deliveries


                Ticket: \"Dear Team,

                My experience with the shipping of order (Order ID: #12751) has been terrible. The package was late, damaged, and the entire process was a hassle. I demand immediate resolution.\"

                Category: Damaged Goods


                Ticket: \"Hello,

                I\'m writing to express my satisfaction with the shipping process for my recent order (Order ID: #77694). The delivery was prompt and the item was well-packaged. Great job!\"

                Category: Customer Support


                Ticket: \"Dear Team,

                The shipping process for my order (Order ID: #72560) was unsatisfactory. There were delays and the package was not handled properly. I\'m not happy with the service provided.\"

                Category: Delayed Deliveries


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the shipping process for my order (Order ID: #8529). The package is lost and the delivery was a nightmare. This is unacceptable and needs to be addressed immediately.\"

                Category: Lost Shipments


                Ticket: \"Dear Customer Service,

                I am extremely impressed with the shipping service for my order (Order ID: #30967). The delivery was exceptionally fast and the package arrived in pristine condition. Kudos to your team for the outstanding service!\"

                Category: Customer Support


                Ticket: \"Hello,

                I had a disappointing experience with the shipping of my recent order (Order ID: #84432). The package was late and arrived in poor condition. I hope this can be improved in the future.\"

                Category: Poor Quality Products


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the shipping of my order (Order ID: #18487). The delivery was lightning fast and the item was perfectly protected. This is by far the best shipping service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Dear Team,

                I wanted to let you know that my order (Order ID: #88132) arrived on time and in perfect condition. I\'m very happy with the shipping service provided. Thank you!\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the shipping process for my order (Order ID: #48968). The package is lost and the delivery was a nightmare. This is unacceptable and needs to be addressed immediately.\"

                Category: Lost Shipments


                Ticket: \"Dear Team,

                I am unhappy with my recent purchase (Order ID: #15295). The product description was inconsistent with the actual item received. This misleading information has led to a very frustrating experience.\"

                Category: Inconsistent Product Descriptions


                Ticket: \"Dear Customer Service,

                I am extremely upset about the misleading product description for my order (Order ID: #50647). The product I received is completely different from what was described. This is false advertising and completely unacceptable. I demand immediate action to resolve this issue.\"

                Category: False Advertising


                Ticket: \"Dear Customer Support,

                I am writing to express my dissatisfaction with the recent order (Order ID: #17658). The product I received was not as described on the website. The description was misleading, and the actual product did not match my expectations. This experience has been very disappointing.\"

                Category: Product Not as Described


                Ticket: \"Dear Team,

                My experience with my recent order (Order ID: #92711) has been terrible. The product description was completely inaccurate, and the item I received is nothing like what was advertised. This false advertising has caused me significant distress, and I demand immediate resolution.\"

                Category: False Advertising


                Ticket: \"Hello,

                I recently placed an order (Order ID: #82485) and was dismayed to find that the product description on your website was inaccurate. The product I received did not match the description at all. This has caused significant inconvenience.\"

                Category: Incorrect Product Details


                Ticket: \"Dear Team,

                My experience with my recent order (Order ID: #82946) has been terrible. The product description was completely inaccurate, and the item I received is nothing like what was advertised. This false advertising has caused me significant distress, and I demand immediate resolution.\"

                Category: False Advertising


                Ticket: \"Dear Customer Service,

                I am extremely upset about the misleading product description for my order (Order ID: #1946). The product I received is completely different from what was described. This is false advertising and completely unacceptable. I demand immediate action to resolve this issue.\"

                Category: False Advertising


                Ticket: \"Dear Customer Support,

                I am writing to express my dissatisfaction with the recent order (Order ID: #87085). The product I received was not as described on the website. The description was misleading, and the actual product did not match my expectations. This experience has been very disappointing.\"

                Category: Product Not as Described


                Ticket: \"Dear Team,

                I am unhappy with my recent purchase (Order ID: #27056). The product description was inconsistent with the actual item received. This misleading information has led to a very frustrating experience.\"

                Category: Inconsistent Product Descriptions


                Ticket: \"Dear Customer Service,

                I am extremely upset about the misleading product description for my order (Order ID: #84078). The product I received is completely different from what was described. This is false advertising and completely unacceptable. I demand immediate action to resolve this issue.\"

                Category: False Advertising


                Ticket: \"Dear Customer Support,
                I recently received my order (Order ID: #95313) and wanted to express my satisfaction with the product quality. The item surpassed my expectations and I am very pleased with my purchase. Thank you for providing such high-quality products!\"

                Category: Customer Support


                Ticket: \"Hello,
                I\'m writing to share my neutral feedback on the product (Order ID: #81397) I received. The quality is okay and meets my basic needs. It\'s neither outstanding nor disappointing.\"

                Category: Customer Support


                Ticket: \"Hello,

                I\'m writing to express my dissatisfaction with the product (Order ID: #49415) I received. The quality is poor and not what I anticipated. I\'m disappointed with the purchase and would like to discuss options for resolution.\"

                Category: Poor Quality Products


                Ticket: \"Dear Customer Service,

                I\'m extremely dissatisfied with the product (Order ID: #19177) I received. The quality is appalling and far below acceptable standards. This experience has been highly disappointing and I expect a prompt resolution.\"

                Category: Poor Quality Products


                Ticket: \"Hello,

                I\'m writing to express my extreme satisfaction with the product (Order ID: #93434) I received. The quality is superb and has exceeded my expectations. Thank you for consistently delivering excellent products.\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I recently received my order (Order ID: #27086) and wanted to provide feedback on the product quality. The item is average in quality and meets basic expectations. While it\'s not exceptional, it serves its purpose adequately.\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I recently received my order (Order ID: #38062) and wanted to express my satisfaction with the product quality. The item surpassed my expectations and I am very pleased with my purchase. Thank you for providing such high-quality products!\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely dissatisfied with the product (Order ID: #66376) I received. The quality is appalling and far below acceptable standards. This experience has been highly disappointing and I expect a prompt resolution.\"

                Category: Poor Quality Products


                Ticket: \"Hello,

                I\'m writing to express my extreme satisfaction with the product (Order ID: #43631) I received. The quality is superb and has exceeded my expectations. Thank you for consistently delivering excellent products.\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I\'m disappointed with the product quality of my recent order (Order ID: #90484). The item does not meet the standards I expected and falls short in terms of quality. I would appreciate assistance with resolving this issue.\"

                Category: Poor Quality Products


                Ticket: \"Hello,

                I\'m writing to express my dissatisfaction with the product (Order ID: #60685) I received. The quality is poor and not what I anticipated. I\'m disappointed with the purchase and would like to discuss options for resolution.\"

                Category: Poor Quality Products


                Ticket: \"Hello,

                I\'m writing to express my extreme satisfaction with the product (Order ID: #40722) I received. The quality is superb and has exceeded my expectations. Thank you for consistently delivering excellent products.\"

                Category: Customer Support


                Ticket: \"Hello,

                I received my order (Order ID: #23071) and I\'m incredibly disappointed with the product quality. The item is unusable due to its extremely poor quality. I request immediate attention to this matter.\"

                Category: Poor Quality Products


                Ticket: \"Dear Customer Support,

                I recently received my order (Order ID: #53908) and wanted to express my satisfaction with the product quality. The item surpassed my expectations and I am very pleased with my purchase. Thank you for providing such high-quality products!\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I recently received my order (Order ID: #95220) and wanted to provide feedback on the product quality. The item is average in quality and meets basic expectations. While it\'s not exceptional, it serves its purpose adequately.\"

                Category: Customer Support


                Ticket: \"Dear Team,

                I want to commend your company for delivering a product (Order ID: #53618) of exceptional quality. It has definitely met and exceeded my expectations. Thank you for your attention to detail and commitment to excellence.\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely dissatisfied with the product (Order ID: #31515) I received. The quality is appalling and far below acceptable standards. This experience has been highly disappointing and I expect a prompt resolution.\"

                Category: Poor Quality Products


                Ticket: \"Dear Customer Service,

                I just received my order (Order ID: #7205) and I\'m blown away by the outstanding quality of the product. It\'s rare to find such high standards, and I wanted to express my appreciation for your dedication to quality.\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I\'m disappointed with the product quality of my recent order (Order ID: #65627). The item does not meet the standards I expected and falls short in terms of quality. I would appreciate assistance with resolving this issue.\"

                Category: Poor Quality Products


                Ticket: \"Dear Customer Support,

                I recently received my order (Order ID: #83586) and wanted to provide feedback on the product quality. The item is average in quality and meets basic expectations. While it\'s not exceptional, it serves its purpose adequately.\"

                Category: Customer Support


                Ticket: \"Hello,

                I received my order (Order ID: #58390) and I\'m incredibly disappointed with the product quality. The item is unusable due to its extremely poor quality. I request immediate attention to this matter.\"

                Category: Poor Quality Products


                Ticket: \"Dear Customer Support,

                I\'m disappointed with the product quality of my recent order (Order ID: #59893). The item does not meet the standards I expected and falls short in terms of quality. I would appreciate assistance with resolving this issue.\"

                Category: Poor Quality Products


                Ticket: \"Dear Customer Support,

                I recently received my order (Order ID: #51079) and wanted to express my satisfaction with the product quality. The item surpassed my expectations and I am very pleased with my purchase. Thank you for providing such high-quality products!\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I just received my order (Order ID: #83183) and I\'m blown away by the outstanding quality of the product. It\'s rare to find such high standards, and I wanted to express my appreciation for your dedication to quality.\"

                Category: Customer Support


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the return process for my order (Order ID: #5052). Everything was seamless, and the customer service support was exceptional. This is the best return experience I\'ve ever had.\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the return process for my order (Order ID: #62504). The entire experience was a nightmare, and I\'m still waiting for my refund. This is unacceptable and needs to be addressed immediately.\"

                Category: Return and Refund Management


                Ticket: \"Hello,

                I had a disappointing experience with the return process for my recent order (Order ID: #6954). The process was not user-friendly, and there were significant delays in processing my refund. This needs improvement.\"

                Category: Return and Refund Management


                Ticket: \"Dear Customer Support,

                I recently went through the return process for my order (Order ID: #46600), and I must say it was smooth and efficient. The refund was processed quickly, and the customer service team was very helpful. Thank you for making this a hassle-free experience!\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the return process for my order (Order ID: #34817). The entire experience was a nightmare, and I\'m still waiting for my refund. This is unacceptable and needs to be addressed immediately.\"

                Category: Return and Refund Management


                Ticket: \"Dear Team,

                The return process for my order (Order ID: #76825) was not satisfactory. I encountered several issues, and it took much longer than expected to resolve. I\'m not happy with the service.\"

                Category: Return and Refund Management


                Ticket: \"Dear Customer Support,

                I recently went through the return process for my order (Order ID: #17369), and I must say it was smooth and efficient. The refund was processed quickly, and the customer service team was very helpful. Thank you for making this a hassle-free experience!\"

                Category: Customer Support


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the return process for my order (Order ID: #81411). Everything was seamless, and the customer service support was exceptional. This is the best return experience I\'ve ever had.\"

                Category: Customer Support


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the return process for my order (Order ID: #26541). Everything was seamless, and the customer service support was exceptional. This is the best return experience I\'ve ever had.\"

                Category: Customer Support


                Ticket: \"Hello,

                I had a disappointing experience with the return process for my recent order (Order ID: #14094). The process was not user-friendly, and there were significant delays in processing my refund. This needs improvement.\"

                Category: Return and Refund Management


                Ticket: \"Dear Team,

                My experience with the return process for order (Order ID: #2889) has been terrible. The process was confusing, and there have been significant delays in receiving my refund. I demand immediate resolution.\"

                Category: Return and Refund Management


                Ticket: \"Dear Customer Support,

                I recently went through the return process for my order (Order ID: #12318), and I must say it was smooth and efficient. The refund was processed quickly, and the customer service team was very helpful. Thank you for making this a hassle-free experience!\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I am extremely impressed with how my return request for order (Order ID: #55069) was handled. The process was incredibly smooth, and the refund was issued faster than expected. Kudos to the team for outstanding service!\"

                Category: Customer Support


                Ticket: \"Hello,

                I\'m writing to express my appreciation for the excellent return process I experienced with my recent order (Order ID: #89285). Everything was handled professionally, and I received my refund promptly. Great job!\"

                Category: Customer Support


                Ticket: \"Dear Team,

                The return process for my order (Order ID: #6843) was not satisfactory. I encountered several issues, and it took much longer than expected to resolve. I\'m not happy with the service.\"

                Category: Return and Refund Management


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the return process for my order (Order ID: #18831). The entire experience was a nightmare, and I\'m still waiting for my refund. This is unacceptable and needs to be addressed immediately.\"

                Category: Return and Refund Management


                Ticket: \"Dear Team,

                I had to return an item from my order (Order ID: #33503), and I was pleasantly surprised by how easy and quick the process was. Thank you for providing such a customer-friendly return policy.\"

                Category: Customer Support


                Ticket: \"Dear Team,

                The return process for my order (Order ID: #55735) was not satisfactory. I encountered several issues, and it took much longer than expected to resolve. I\'m not happy with the service.\"

                Category: Return and Refund Management


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the return process for my order (Order ID: #41814). Everything was seamless, and the customer service support was exceptional. This is the best return experience I\'ve ever had.\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the return process for my order (Order ID: #57088). The entire experience was a nightmare, and I\'m still waiting for my refund. This is unacceptable and needs to be addressed immediately.\"

                Category: Return and Refund Management


                Ticket: \"Dear Team,

                My experience with your customer service regarding order (Order ID: #22308) has been terrible. The representative was extremely rude and unhelpful, leaving my issue unresolved and causing me significant distress. I demand immediate action to rectify this situation.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Hello,

                I had a disappointing experience with your customer service team regarding my order (Order ID: #20797). The representative I spoke to was very rude and did not offer any helpful solutions. This needs to be addressed.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Team,

                I am unhappy with the way my issue was handled by your customer service representative regarding order (Order ID: #71528). The representative was unprofessional and unhelpful, leaving me frustrated and dissatisfied.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Hello,

                I am writing to express my extreme dissatisfaction with the customer service I received regarding my order (Order ID: #91130). The representative was incredibly rude and did not provide any assistance. This experience has been extremely frustrating and disappointing.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Team,

                My experience with your customer service regarding order (Order ID: #42062) has been terrible. The representative was extremely rude and unhelpful, leaving my issue unresolved and causing me significant distress. I demand immediate action to rectify this situation.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Team,

                I am unhappy with the way my issue was handled by your customer service representative regarding order (Order ID: #25549). The representative was unprofessional and unhelpful, leaving me frustrated and dissatisfied.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Customer Service,

                I am extremely upset with the treatment I received from your customer service representative regarding my order (Order ID: #10462). The representative was not only rude but also completely unhelpful and dismissive. This is completely unacceptable and needs immediate attention.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Team,

                I am unhappy with the way my issue was handled by your customer service representative regarding order (Order ID: #71806). The representative was unprofessional and unhelpful, leaving me frustrated and dissatisfied.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Customer Service,

                I am extremely upset with the treatment I received from your customer service representative regarding my order (Order ID: #29574). The representative was not only rude but also completely unhelpful and dismissive. This is completely unacceptable and needs immediate attention.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Customer Support,

                I\'m writing to express my frustration with the interaction I had with one of your representatives regarding my order (Order ID: #83657). The representative was rude and unhelpful, making the entire experience very unpleasant. This level of service is not acceptable.\"

                Category: Rude or Unhelpful Representatives


                Ticket: \"Dear Team,

                The technical issue with my order (Order ID: #52166) was handled poorly. The process was slow and frustrating, and I\'m not happy with the service provided.\"

                Category: Technical Issues


                Ticket: \"Dear Team,

                My experience with the technical issue related to order (Order ID: #35878) has been terrible. The process was confusing, and there have been significant delays in resolving the issue. I demand immediate resolution.\"

                Category: Technical Issues


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the resolution of the technical issue related to my order (Order ID: #98734). The support was top-notch, and the issue was fixed immediately. This is by far the best customer service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I recently experienced a technical issue with my order (Order ID: #40072). The resolution process was average, neither exceptionally good nor bad. The issue was eventually resolved, but it took a bit of time.\"

                Category: Technical Issues


                Ticket: \"Dear Team,

                I had a technical issue with my order (Order ID: #66446), but the support team handled it very well. The problem was resolved swiftly, and I appreciate the excellent service provided.\"

                Category: Customer Support


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the technical issue I faced with my order (Order ID: #61230). The entire experience was a nightmare, and the issue is still not fully resolved. This is unacceptable and needs to be addressed immediately.\"

                Category: Technical Issues


                Ticket: \"Dear Customer Service,

                I am extremely impressed with the way the technical issue with my order (Order ID: #58223) was handled. The support team was exceptionally quick and professional, and the issue was resolved in no time. Kudos to your team for outstanding service!\"

                Category: Customer Support


                Ticket: \"Hello,

                I\'m writing to share my neutral feedback on the technical issue I faced with my order (Order ID: #31530). The support was adequate, and the issue was resolved, but it wasn\'t a particularly memorable experience.\"

                Category: Technical Issues


                Ticket: \"Dear Customer Support,

                I\'m writing to express my frustration with the technical issue I encountered with my order (Order ID: #61598). The resolution process was slow, and it took a lot of effort to get the issue fixed. This experience has been quite disappointing.\"

                Category: Technical Issues


                Ticket: \"Dear Customer Support,

                I recently experienced a technical issue with my order (Order ID: #16306), and I was pleasantly surprised by how quickly the issue was resolved. The support team was very helpful and efficient. Thank you for the great service!\"

                Category: Customer Support


                Ticket: \"Dear Team,

                I had a technical issue with my order (Order ID: #84666), but the support team handled it very well. The problem was resolved swiftly, and I appreciate the excellent service provided.\"

                Category: Customer Support


                Ticket: \"Dear Team,

                My experience with the technical issue related to order (Order ID: #39588) has been terrible. The process was confusing, and there have been significant delays in resolving the issue. I demand immediate resolution.\"

                Category: Technical Issues


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the resolution of the technical issue related to my order (Order ID: #32922). The support was top-notch, and the issue was fixed immediately. This is by far the best customer service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Hello,

                I\'m writing to share my neutral feedback on the technical issue I faced with my order (Order ID: #65296). The support was adequate, and the issue was resolved, but it wasn\'t a particularly memorable experience.\"

                Category: Technical Issues


                Ticket: \"Dear Customer Support,

                I\'m writing to express my frustration with the technical issue I encountered with my order (Order ID: #37027). The resolution process was slow, and it took a lot of effort to get the issue fixed. This experience has been quite disappointing.\"

                Category: Technical Issues


                Ticket: \"Dear Team,

                My experience with the technical issue related to order (Order ID: #28446) has been terrible. The process was confusing, and there have been significant delays in resolving the issue. I demand immediate resolution.\"

                Category: Technical Issues


                Ticket: \"Hello,

                I\'m writing to express my appreciation for the prompt resolution of the technical issue I encountered with my recent order (Order ID: #12676). The support team was excellent and resolved the issue quickly. Great job!\"

                Category: Customer Support


                Ticket: \"Hello,

                I\'m writing to share my neutral feedback on the technical issue I faced with my order (Order ID: #8722). The support was adequate, and the issue was resolved, but it wasn\'t a particularly memorable experience.\"

                Category: Technical Issues


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the resolution of the technical issue related to my order (Order ID: #22566). The support was top-notch, and the issue was fixed immediately. This is by far the best customer service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I\'m writing to express my frustration with the technical issue I encountered with my order (Order ID: #20044). The resolution process was slow, and it took a lot of effort to get the issue fixed. This experience has been quite disappointing.\"

                Category: Technical Issues


                Ticket: \"Hello,

                I had a disappointing experience with the technical issue related to my recent order (Order ID: #63501). The support was not very helpful, and there were significant delays in resolving the issue. I hope this can be improved in the future.\"

                Category: Technical Issues


                Ticket: \"Dear Customer Service,

                I\'m extremely frustrated with the technical issue I faced with my order (Order ID: #19522). The entire experience was a nightmare, and the issue is still not fully resolved. This is unacceptable and needs to be addressed immediately.\"

                Category: Technical Issues


                Ticket: \"Hello,

                I wanted to share my wonderful experience with the resolution of the technical issue related to my order (Order ID: #16540). The support was top-notch, and the issue was fixed immediately. This is by far the best customer service I have ever experienced.\"

                Category: Customer Support


                Ticket: \"Dear Customer Support,

                I recently experienced a technical issue with my order (Order ID: #89901), and I was pleasantly surprised by how quickly the issue was resolved. The support team was very helpful and efficient. Thank you for the great service!\"

                Category: Customer Support


                Ticket: \"Hello,

                I\'m writing to share my neutral feedback on the technical issue I faced with my order (Order ID: #87443). The support was adequate, and the issue was resolved, but it wasn\'t a particularly memorable experience.\"

                Category: Technical Issues

                Ticket: {text}
                Category:
                """
                response = self.model.predict(prompt, **self.parameters)
                category = response.text.strip()
                element['Category'] = category
            else:
                logger.warning("No 'EmailText' found in the element")
                element['Category'] = 'Unresolved Issues'  # Default value
            yield element
        except Exception as e:
            logger.error(f"Error in AnalyzeClassification: {e}")
            element['Category'] = 'Unresolved Issues'  # Default value
            yield element

class AnalyzeOrderId(beam.DoFn):
    def __init__(self, project, location, model_name):
        self.project = project
        self.location = location
        self.model_name = model_name
    #     # self.bucket_name = bucket_name

    def setup(self):
        import vertexai
        from vertexai.language_models import TextGenerationModel

        vertexai.init(project=self.project, location=self.location)
        self.model = TextGenerationModel.from_pretrained(self.model_name)
        self.parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.8
            }
    

    def process(self, element):
        import json

        # for record in element:
        # text = record.get('EmailBody', '')
        try:
            text = element.get('EmailText', '')
            if text:
                prompt = f"""
                Extract the order Id  from the input text. Do not give previous id\'s.
                Order Id start with after # followed by only numbers. Only print the order id, nothing else.
                Don\'t give me in JSON Format or other format. Just give me in plain text like below:
                only print one order id which is given in the input text.

                Text: \"Dear Customer Support,

                I recently received my order (Order ID: #90680) and I am pleased to report that the shipping process was smooth. The delivery was timely and the package arrived in excellent condition. Thank you for the great service!\"

                JSON: #90680



                Text: \"Dear Customer Support,
                I\\\'m writing to express my frustration with the shipping process for my order (Order ID: #31527). The delivery was delayed and the package arrived damaged. This experience has been quite disappointing.\"

                JSON: #31527


                Text: \"Dear Customer Service,

                I am extremely impressed with the shipping service for my order (Order ID: #72836). The delivery was exceptionally fast and the package arrived in pristine condition. Kudos to your team for the outstanding service!\"

                JSON: #72836


                Text: \"Hello,
                I ordered shoes with id #38384 for pincode 484848 and amount 388500. I have raised the ticket about my complaint with ticket id 34.\"

                JSON: #38384


                Text: \" Hi Customer Support, 
                I have issue with my order (#8790). Please help me into this my mobile number is 7878787900 and my zip code is 567890 \"

                JSON: #8790


                Text: \"Dear Customer Support,
                I have ordered (#45678) from your platform but I don\'t like the product because I ordered the different product but received different one.\"

                JSON: #45678
                                

                Text:{text}

                JSON:
                """
                logging.info(f"Prompt sent to model: {prompt}")
                response = self.model.predict(prompt, **self.parameters)
                logging.info(f"Model response: {response.text}")

                order_id = response.text.strip()
                if not order_id:
                    order_id = 'N/A'
                
                element['OrderID'] = order_id
            else:
                logging.warning("No 'EmailText' found in the element")
                element['OrderID'] = 'N/A'
            
            yield element
        except Exception as e:
            logging.error(f"Error in AnalyzeOrderId: {e}")
            element['OrderID'] = 'N/A'
            yield element

# def run():
#     project = 'ai-ml-solutions'
#     bucket_name = 'email_content_files'
#     output_table = 'ai-ml-solutions:Anomaly_detection.sentiment_category'

#     options = PipelineOptions(
#         # runner='DataflowRunner',
#         project=project,
#         region='us-east1',
#         job_name='analyze-sentiment-dataflow-pipeline'
#     )

#     with beam.Pipeline(options=options) as p:
#         (p
#          | 'MatchFiles' >> fileio.MatchFiles("gs://email_content_files/*.json")
#          | 'ReadMatches' >> fileio.ReadMatches()
#          | 'AnalyzeSentiment' >> beam.ParDo(AnalyzeSentiment(project, 'us-central1', 'text-bison@002', bucket_name))
#          | 'AnalyzeCategory' >> beam.ParDo(AnalyzeClassification(project, 'us-central1', 'text-bison', bucket_name))
#          | 'AnalyzeOrder' >> beam.ParDo(AnalyzeOrderId(project, 'us-central1', 'text-bison', bucket_name))
         #| beam.Map(print)
        #  | 'WriteToBigQuery' >> WriteToBigQuery(
        #         "ai-ml-solutions.Anomaly_detection.Test_table",
        #         # schema='CreatedTime: TIMESTAMP , EmailText:str, subject: str , sentiment:str,category:str, order_id:str',
        #         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        #         create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
        #     )

    #      | 'WriteToBigQuery' >> WriteToBigQuery(
    #             "ai-ml-solutions.Anomaly_detection.Output_Table",
    #             # schema='TimeStamp: TIMESTAMP , OrderID:str, Category: str , SubCategory:str,Sentiment:str, EmailText:str',
    #             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
    #             #create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
    #         )

    # )

class ParseJsonFn(beam.DoFn):
    def process(self, element):
        try:
            # Since NDJSON is JSON objects separated by newlines, split by newline
            lines = element.splitlines()
            for line in lines:
                if line.strip():  # Skip empty lines
                    yield json.loads(line)
        except Exception as e:
            logging.error(f"Error parsing JSON: {e}")

def format_result(element):
    if isinstance(element, dict):
        formatted_result = {
            'TimeStamp': element.get('TimeStamp', 'N/A'),
            'OrderID': element.get('OrderID', 'N/A'),
            'Category': element.get('Category', 'N/A'),
            'Sentiment': element.get('Sentiment', 'Neutral'),
            'EmailText': element.get('EmailText', 'N/A')
        }
        logging.info(f"Formatted result: {formatted_result}")
        return formatted_result
    else:
        logging.error(f"Unexpected element type: {type(element)}. Element: {element}")
        return None

class ParseTimestamp(beam.DoFn):
    def process(self, element):
        logger.info(f"Processing element: {element}")
        if not isinstance(element, dict):
            logger.error(f"Expected a dictionary but got: {type(element)}")
            return
        timestamp_str = element.get('TimeStamp', '')
        try:
            timestamp = dt.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f UTC')
            element['timestamp'] = timestamp
            yield element
        except ValueError as e:
            logger.error(f"Invalid timestamp format: {timestamp_str}, error: {e}")

class TryExceptFilter(beam.DoFn):
    def process(self, element, time_window_minutes):
        try:
            if element['timestamp'] > dt.utcnow() - timedelta(minutes=time_window_minutes):
                yield element
        except KeyError as e:
            logger.error(f"KeyError in element {element}: {e}")
        except TypeError as e:
            logger.error(f"TypeError in element {element}: {e}")

def run():
    project = 'golden-medium-427507-p9'
    bucket_name = 'dataflow-test-14'
    pubsub_subscription = f'projects/{project}/subscriptions/crm_sb'
    output_table = f'{project}:Demo.Anomaly_Detection'

    pipeline_options = PipelineOptions(
        flags=[
            f'--project={project}',
            f'--temp_location=gs://{bucket_name}/temp/',
            f'--staging_location=gs://{bucket_name}/staging/',
            '--runner=DataflowRunner',
            '--region=us-west1',
            '--worker_machine_type=n1-standard-4',
            '--num_workers=5',
            '--max_num_workers=10',
            '--autoscaling_algorithm=THROUGHPUT_BASED',
            '--requirements_file=requirements.txt'
        ],
        streaming=True,
        save_main_session=True
    )
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    
    try:
        with beam.Pipeline(options=pipeline_options) as p:
            (p
                | 'Read from Pub/Sub' >> beam.io.ReadFromPubSub(subscription=pubsub_subscription)
                | 'Parse NDJSON' >> beam.ParDo(ParseJsonFn())
                | 'Filter Invalid Elements' >> beam.Filter(lambda x: x is not None)
                | 'Parse Timestamp' >> beam.ParDo(ParseTimestamp())
                | 'Filter Current Data' >> beam.ParDo(TryExceptFilter(), time_window_minutes=15)
                | 'ApplyFixedWindow' >> beam.WindowInto(
                        FixedWindows(15 * 60),  # 15 minutes
                        trigger=AfterWatermark(
                            early=AfterProcessingTime(5 * 60),
                            late=AfterProcessingTime(10 * 60)
                        ),
                        accumulation_mode=AccumulationMode.DISCARDING
                    )
                | 'Analyze Sentiment' >> beam.ParDo(AnalyzeSentiment(project, 'us-central1', 'text-bison@002'))
                | 'Analyze Category' >> beam.ParDo(AnalyzeClassification(project, 'us-central1', 'gemini-1.5-flash-001'))
                | 'Analyze Order ID' >> beam.ParDo(AnalyzeOrderId(project, 'us-central1', 'gemini-1.5-flash-001'))
                #| 'Map to KV' >> beam.Map(lambda x: (x['OrderID'], x))
                #| 'Combine by Key' >> beam.CombinePerKey(lambda elements: list(elements))
                #| 'Count Events Globally' >> beam.combiners.Count.Globally().without_defaults()
                | 'Format Results' >> beam.Map(format_result)
                | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                        table=output_table,
                        schema='TimeStamp:TIMESTAMP,OrderID:STRING,Category:STRING,Sentiment:STRING,EmailText:STRING',
                        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
                    )
            )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

if __name__ == '__main__':
    run()
