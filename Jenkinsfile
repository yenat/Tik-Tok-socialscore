pipeline {
    agent any
    environment {
        DOCKER_IMAGE = 'tiktok-socialscore-api'
        DOCKER_TAG = 'latest'
        API_URL = 'http://192.168.20.49:8000'
        CALLBACK_URL = 'http://192.168.20.49:9696/customer/social-score'
        TEST_USERNAME = 'testuser'
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
                    // Ensure jq is installed for JSON parsing
                    sh 'apt-get update && apt-get install -y jq || true'
                    
                    // Clean up existing container
                    sh 'docker stop ${DOCKER_IMAGE} || true'
                    sh 'docker rm ${DOCKER_IMAGE} || true'
                    
                    // Start container with logs visible
                    sh """
                    docker run -d \\
                        --name ${DOCKER_IMAGE} \\
                        -p 8000:8000 \\
                        ${DOCKER_IMAGE}:${DOCKER_TAG}
                    """
                    
                    // Wait for container to start
                    sleep(10)
                    
                    // Check container status
                    def status = sh(returnStdout: true, script: "docker inspect -f '{{.State.Status}}' ${DOCKER_IMAGE}").trim()
                    if (status != "running") {
                        sh "docker logs ${DOCKER_IMAGE}"
                        error("Container failed to start. Status: ${status}")
                    }
                    
                    // Verify API health
                    def healthCheck = sh(returnStdout: true, script: "curl -s -f ${API_URL}/health || echo 'FAILED'").trim()
                    if (healthCheck == 'FAILED') {
                        sh "docker logs ${DOCKER_IMAGE}"
                        error("API health check failed")
                    }
                }
            }
        }
        
        stage('Test API') {
            steps {
                script {
                    // Generate unique test ID
                    def TEST_ID = "jenkins-test-${BUILD_NUMBER}"
                    
                    // 1. Request Verification Code
                    def verificationResponse = sh(returnStdout: true, script: """
                        curl -s -X POST '${API_URL}/request-verification' \\
                        -H 'Content-Type: application/json' \\
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "type": "*SOCIAL_SCORE*",
                            "data": [{
                                "social_media": "TikTok",
                                "username": "${TEST_USERNAME}"
                            }]
                        }'
                    """).trim()
                    
                    echo "Verification response: ${verificationResponse}"
                    
                    // Verify response contains verification code
                    if (!verificationResponse.contains('verification_code')) {
                        error("Verification request failed: ${verificationResponse}")
                    }
                    
                    // Parse verification code
                    def verificationCode = sh(returnStdout: true, script: """
                        echo '${verificationResponse}' | jq -r '.verification_code'
                    """).trim()
                    
                    echo "Verification code: ${verificationCode}"
                    
                    // 2. Mock verification by updating test storage (since we can't modify real TikTok bio)
                    // This assumes your API has a way to mock the verification for testing
                    // Alternatively, you could use a test TikTok account you control
                    def mockVerify = sh(returnStdout: true, script: """
                        curl -s -X POST '${API_URL}/mock-verify' \\
                        -H 'Content-Type: application/json' \\
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "code": "${verificationCode}"
                        }'
                    """).trim()
                    
                    echo "Mock verify response: ${mockVerify}"
                    
                    // 3. Verify and score with callback
                    def scoreResponse = sh(returnStdout: true, script: """
                        curl -v -X POST '${API_URL}/verify-and-score' \\
                        -H 'Content-Type: application/json' \\
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "type": "*SOCIAL_SCORE*",
                            "data": [{
                                "social_media": "TikTok",
                                "username": "${TEST_USERNAME}"
                            }],
                            "callbackUrl": "${CALLBACK_URL}"
                        }'
                    """).trim()
                    
                    echo "Score response: ${scoreResponse}"
                    
                    // Verify score response
                    if (!scoreResponse.contains('socialscore')) {
                        error("Score request failed: ${scoreResponse}")
                    }
                    
                    // 4. Verify callback was triggered (mock check)
                    echo "MOCK: Would verify callback was sent to ${CALLBACK_URL}"
                }
            }
        }
    }
    post {
        always {
            script {
                echo "Collecting container logs..."
                sh 'docker logs ${DOCKER_IMAGE} --tail 100'
                sh 'docker stop ${DOCKER_IMAGE} || true'
                sh 'docker rm ${DOCKER_IMAGE} || true'
            }
        }
    }
}