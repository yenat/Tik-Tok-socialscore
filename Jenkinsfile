pipeline {
    agent any
    environment {
        DOCKER_IMAGE = 'tiktok-socialscore-api'
        DOCKER_TAG = 'latest'
        API_URL = 'http://localhost:8000'
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
                    
                    // Start new container with restart policy
                   sh """
                        docker run -d \
                            --name ${DOCKER_IMAGE} \
                            -p 8000:8000 \
                            --restart unless-stopped \
                            -e API_HOST=0.0.0.0 \  
                            ${DOCKER_IMAGE}:${DOCKER_TAG}
                    """
                    
                    // Extended health check with container status verification
                    def healthy = false
                    for (int i = 0; i < 10; i++) {
                        // First check if container is running
                        def containerRunning = sh(
                            returnStdout: true,
                            script: "docker inspect -f '{{.State.Running}}' ${DOCKER_IMAGE} || echo 'false'"
                        ).trim()
                        
                        if (containerRunning != "true") {
                            echo "Container is not running"
                            break
                        }
                        
                        // Then check API health
                        def status = sh(returnStatus: true, 
                            script: "curl -s -f ${API_URL}/health > /dev/null")
                        
                        if (status == 0) {
                            healthy = true
                            break
                        }
                        sleep(5)
                    }
                    
                    if (!healthy) {
                        // Get detailed logs before failing
                        sh "docker logs ${DOCKER_IMAGE} --tail 100"
                        error("API failed to start or container stopped running")
                    } else {
                        echo "API deployed successfully and healthy"
                        // Verify container remains running
                        sh """
                            echo "Container status:"
                            docker ps --filter "name=${DOCKER_IMAGE}"
                        """
                    }
                }
            }
        }
    }
    post {
        always {
            script {
                // Output container status and logs
                sh """
                    echo "### Container Status ###"
                    docker ps -a --filter "name=${DOCKER_IMAGE}"
                    echo "\\n### Last 100 Logs ###"
                    docker logs ${DOCKER_IMAGE} --tail 100 || true
                """
            }
        }
    }
}