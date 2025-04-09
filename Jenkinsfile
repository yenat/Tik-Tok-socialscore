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
                    // Stop and remove any existing container
                    sh 'docker stop ${DOCKER_IMAGE} || true'
                    sh 'docker rm ${DOCKER_IMAGE} || true'
                    
                    // Start new container
                    sh "docker run -d --name ${DOCKER_IMAGE} -p 8000:8000 ${DOCKER_IMAGE}:${DOCKER_TAG}"
                    sleep(10)
                    
                    // Health check with retries
                    def healthy = false
                    for (int i = 0; i < 5; i++) {
                        def status = sh(returnStatus: true, 
                            script: "curl -s -f ${API_URL}/health > /dev/null")
                        if (status == 0) {
                            healthy = true
                            break
                        }
                        sleep(5)
                    }
                    if (!healthy) {
                        error("API failed to start")
                    } else {
                        echo "API deployed successfully and healthy"
                    }
                }
            }
        }
    }
    post {
        always {
            script {
                // Output the last 50 lines of logs for debugging if needed
                sh 'docker logs ${DOCKER_IMAGE} --tail 50'
            }
        }
    }
}