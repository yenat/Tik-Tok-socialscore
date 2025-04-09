pipeline {
    agent any
    environment {
        DOCKER_IMAGE = 'tiktok-socialscore-api'
        DOCKER_TAG = 'latest'
        API_URL = 'http://192.168.20.49:8000'
    }
    stages {
        stage('Checkout') {
            steps { 
                git url: 'https://github.com/yenat/Tik-Tok-socialscore.git', branch: 'main' 
            }
        }
        
        stage('Build Docker') {
            steps { 
                script { 
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}") 
                } 
            }
        }
        
        stage('Deploy API') {
            steps {
                script {
                    // Clean up any existing container
                    sh 'docker stop ${DOCKER_IMAGE} || true'
                    sh 'docker rm ${DOCKER_IMAGE} || true'
                    
                    // Start new container with explicit keep-alive
                    sh """
                        docker run -d \
                            --name ${DOCKER_IMAGE} \
                            -p 8000:8000 \
                            ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            /bin/sh -c "uvicorn app:app --host 0.0.0.0 --port 8000"
                    """
                    
                    // Verify container stays running
                    sleep(10)
                    def running = sh(
                        returnStdout: true,
                        script: "docker inspect -f '{{.State.Running}}' ${DOCKER_IMAGE}"
                    ).trim()
                    
                    if (running != "true") {
                        sh "docker logs ${DOCKER_IMAGE}"
                        error("Container failed to stay running")
                    }
                }
            }
        }
    }
}