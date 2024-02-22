---
layout: post
title:  "Talk to your Notion with OpenAI Functions"
date:   2024-02-21 18:37:21 +0300
class: post-template
subclass: 'post'
comments: true
category: education
tag: openai, language models, retrieval augmented generation
---

![Robot and human talking](/assets/images/notion-openai/Robot-and-human-talking.png)

[**_Retrieval Augmented Generation_**](https://arxiv.org/abs/2312.10997) (_or RAG) has proven to be one of the most substantial advancements in the practical application of Large Language Models. It allows providing the necessary context to your conversational agents, practically tailoring them to your needs. Moreover, it only requires a little machine learning competence to implement, and it is transparent for a non-expert user._

_Recently, OpenAI and other LLM providers released_ [_function calling_](https://platform.openai.com/docs/guides/function-calling)_: a way for an LLM to call additional actions when necessary to answer your questions. It can be an alternative to RAG, as you may use functions to provide personalized context from external resources. I will explain how to build a Notion assistant using it in this article._

First, I absolutely love [Notion](https://www.notion.so/): I use it to keep all my notes, and it has grown into an extensive personal database. I keep there my diary, ideas, education notes, projects and many other things. I invested a lot of time into organizing it, and I still can’t find the perfect structure. So when I asked myself, “How can I use LLMs to improve my life?” my first idea was to make it help me with my Notion organization.

As many other services, Notion [offers an API](https://developers.notion.com/) that you can use to search pages and retrieve data from them. And said API can be called as a function by an OpenAI assistant when it requires some additional context to help you. Structured API response (JSON in Notion’s case) will augment response generation, just as text chunks augment it in RAG. So practically, you will talk to your assistant, and it will, in turn, talk to Notion.

You can implement such a conversational agent with Python and Jupyter Notebook. I will explain the idea using [this example](https://colab.research.google.com/drive/1ZHDN3Aj5Ms35D_njBZqk6V70u792dwwj#scrollTo=VlMtMURdRqNu) I did in Google Collab. And I will share a code snippet you can utilize for an end-to-end conversation loop.

# Secrets

First, we need to configure programmatic access to OpenAI and Notion.

For **OpenAI**, go to [API keys](https://platform.openai.com/api-keys) page and generate a new key. 

For **Notion,** go to [My integrations](https://www.notion.so/my-integrations) page. Here, you need to create an integration associated with the workspace you plan to use. Get the integration token for your integration. You also need to explicitly connect all the pages you want to access with your integration:

![Connect a notion page](/assets/images/notion-openai/Notion-connect.png)

If you want your assistant to access all your pages, connect your integration to all parent pages in the workspace. It will extend the integration connection to child access.

Now, you can use API tokens for programmatic access.

# Function calling workflow

First, implement the functions you are going to call:

```python
def search_my_notion(query: str):
  return notion.search(query=query, page_size=10)

def get_notion_page_content(page_id: str):
  return notion.blocks.children.list(page_id)
```

Next, create a new assistant in OpenAI that can access the functions:

```python
from openai import OpenAI
client = OpenAI(api_key=openai_token)

tools = [
  {
    "type": "function",
    "function": {
      "name": "search_my_notion",
      "description": "Search Notion pages and databases of the user",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Query to search over titles",
          },
        },
        "required": ["query"],
      },
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_notion_page_content",
      "description": "Obtain notion page content as blocks",
      "parameters": {
        "type": "object",
        "properties": {
          "page_id": {
            "type": "string",
            "description": "ID of a page",
          },
        },
        "required": ["page_id"],
      },
    }
  }
]

name = "Notion Assistant"
notion_assistant = client.beta.assistants.create(
    instructions="""
Answer informally, but politely. Use Notion API access as needed. Say hello to the user in the first message using their name.
""",
    name=name,
    tools=tools,
    model="gpt-4-turbo-preview",
)
```

Let's now create a thread and run it:

```python
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Hi! I am looking for my notes about RAG, can you show them to me?"
)
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=notion_assistant.id,
  instructions="Address the user as Anatolii."
)
```

`[run.status](https://platform.openai.com/docs/api-reference/runs/object#runs/object-status)` indicates the state of an async task execution in OpenAI. You can refresh it:

```python
run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)
print(run.status)
```

We should do nothing while the status is `in_progress`. If it turns to **`requires_action` we need to perform all actions listed in `run.required_action`:**

```python
tool_outputs = []
for tool_call in run.required_action.submit_tool_outputs.tool_calls:
  function_name = tool_call.function.name
  function = globals().get(function_name)
  arguments = json.loads(tool_call.function.arguments)
  tool_call_result = function(**arguments)
  tool_outputs.append({
    "tool_call_id": tool_call.id,
    "output": str(tool_call_result)
  })

client.beta.threads.runs.submit_tool_outputs(
  thread_id=thread.id,
  run_id=run.id,
  tool_outputs=tool_outputs
)
```

Now, we should wait again for the run execution. It may request more actions, but eventually, the status will be `completed.` After that, you can see the model response:

```python
messages = client.beta.threads.messages.list(
  thread_id=thread.id
)
print(essages.data[0].content[0].text.value)
```

# End-to-end example

For this experiment, I have prepared a ready-to-use code snippet for a chat loop, which you can use in Jupyter Notebook:

```python
import time
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads.run import Run, RequiredAction
import IPython
import json
import traceback
from openai import OpenAI
from notion_client import Client

notion = Client(auth=notion_token)
client = OpenAI(api_key=openai_token)

notion_assistant = client.beta.assistants.retrieve("asst_fPOHvsq7u9Rqrxe1KKNgFNCw")

def wait_for_run(run: Run, thread: Thread) -> Run:
  inprogress_statuses = ["queued", "in_progress"]
  failed_statuses = ["failed", "expired", "cancelling", "cancelled"]

  while True:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(
      thread_id=thread.id,
      run_id=run.id
    )
    print(run.status)
    if run.status not in inprogress_statuses:
      break

  if run.status in failed_statuses:
    raise Exception(f"OpenAI run can not be completed: {run.status}, {run.last_error}")

  return run

def perform_action(run: Run, thread: Thread) -> Run:
  if run.required_action.type != "submit_tool_outputs":
    raise Exception(f"Unknown action type: {run.required_action.type}")

  # Get outputs for all requested tools
  tool_outputs = []
  for tool_call in run.required_action.submit_tool_outputs.tool_calls:
    if tool_call.type != "function":
      raise Exception(f"Unknown tool call type: {tool_call.type}")

    function_name = tool_call.function.name
    function = globals().get(function_name)
    if not function:
      raise Exception(f"Function {function_name} not found")
    arguments = json.loads(tool_call.function.arguments)
    tool_call_result = function(**arguments)
    tool_outputs.append({
      "tool_call_id": tool_call.id,
      "output": str(tool_call_result)
    })

  # Submit them to OpenAI
  client.beta.threads.runs.submit_tool_outputs(
    thread_id=thread.id,
    run_id=run.id,
    tool_outputs=tool_outputs
  )

  # After that the only expected status is completed
  run = wait_for_run(run, thread)
  match run.status:
    case "completed":
      return run
    case "requires_action":
      return perform_action(run, thread)
    case _:
      raise Exception(f"Unexpected run status: {run.status}, {run.last_error}")

  return run

def receive_run_response(run: Run, thread: Thread) -> str:
  run = wait_for_run(run, thread)
  if run.status == "requires_action":
    run = perform_action(run, thread)

  messages = client.beta.threads.messages.list(
    thread_id=thread.id
  )
  return messages.data[0].content[0].text.value

def display_conversation(conversation: list[str, str]) -> None:
  display_objects = []

  # Show only ten
  for role, message in conversation[:10]:
    if role == "user":
      message_to_display = f"**You**\n\n{message}"
    if role == "assistant":
      message_to_display = f"**Assistant**\n\n{message}"
    display_object = IPython.display.Markdown(message_to_display)
    display_objects.append(display_object)

  IPython.display.display(*display_objects, clear=True)

def conversation_loop(assistant) -> None:
  conversation = []
  thread = client.beta.threads.create()
  while True:
    # Get message from a user
    message = input()
    if message == "exit":
      break

    conversation.append(("user", message))
    message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=message
    )
    display_conversation(conversation)

    # Start new run and receive a response
    run = client.beta.threads.runs.create(
      thread_id=thread.id,
      assistant_id=assistant.id,
      additional_instructions="Address the user Anatolii."
    )
    response = receive_run_response(run, thread)
    conversation.append(("assistant", response))
    display_conversation(conversation)

try:
  conversation_loop(notion_assistant)
except Exception as e:
  print(traceback.format_exc())
```

With it, you will have a simple chat interface like this:

![Untitled](/assets/images/notion-openai/Chat-example.png)

You can also add more functions. My example has only a few read functions, but if you add methods to create or update pages, your assistant can be a great help. Have fun!

---

Language models are a powerful tool, but one task they will not handle is defending Ukraine from Russian aggression. Ukrainian Armed Forces can handle that, and we are helping UAF to handle that. Right now, I am supporting **PVP section**: a group of engineers engaged in the construction of FPV drones, which are proven to be a simple and powerful resource for aerial reconnaissance or kamikaze attacks. You can make a one-time or regular donation to the PVP section [here](https://www.buymeacoffee.com/pvp_section/jotocekilu). 
