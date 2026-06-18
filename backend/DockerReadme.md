1.Install Docker Desktop and make sure it is running
2.Open terminal and go to your project folder
    cd your-project-folder
3.Create a requirements.txt file with your dependencies
4.Create a Dockerfile in the project
5.(Optional) Create a .env file for API keys
6.Build the Docker image
    docker build -t chatbot .
7.Run the Docker container
    docker run -p 8000:8000 --env-file .env chatbot
8.Open your browser and go to
    http://localhost:8000/docs