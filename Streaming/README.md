#🚀 Execution Steps
#Create PDF Folder: 
Ensure that a folder named runbooks_pdf exists in your project root folder. Place some PDF runbooks related to your cloud/Kubernetes inside it.
Start the Server: Run the following command in your terminal:Bash

python3 -m uvicorn main:app --reload

(The server will successfully start running at [http://127.0.0.1:8000](http://127.0.0.1:8000)).
How to Test Real-Time Streaming? (Testing Stream):You can call the /chat/stream (POST Method) endpoint using Postman, 
cURL, or a small HTML/JS frontend.To test using cURL:Bash

curl -N -X POST "http://127.0.0.1:8000/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "How to fix Kubernetes OOMKilled error?", "thread_id": "session_1"}'
When you run this, you will receive the live status of each agent (rag_search $\rightarrow$ draft_solution $\rightarrow$ security_review) in real-time!Admin Approval:After the agent pauses for approval, you can complete the workflow by sending JSON data to the /approve (POST Method) endpoint as follows:JSON{
  "thread_id": "session_1",
  "approved": true
}
