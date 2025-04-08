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
                    // Ensure jq is available for JSON parsing
                    def jqCheck = sh(returnStatus: true, script: 'command -v jq')
                    if (jqCheck != 0) {
                        // Try multiple installation methods
                        sh '''
                            (apt-get update -qq && apt-get install -y jq) || \
                            (yum install -y jq) || \
                            (wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64 && \
                            chmod +x jq && mv jq /usr/local/bin/)
                        '''
                    }
                    
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
                    
                    // Parse verification code with jq
                    def verificationCode = sh(returnStdout: true, script: """
                        echo '${verificationResponse}' | jq -r '.verification_code'
                    """).trim()
                    
                    echo "Verification code: ${verificationCode}"
                    
                    // 2. Mock verification (if endpoint exists)
                    try {
                        def mockVerify = sh(returnStdout: true, script: """
                            curl -s -X POST '${API_URL}/mock-verify' \\
                            -H 'Content-Type: application/json' \\
                            -d '{
                                "fayda_number": "${TEST_ID}",
                                "code": "${verificationCode}"
                            }'
                        """).trim()
                        echo "Mock verify response: ${mockVerify}"
                    } catch (Exception e) {
                        echo "Mock verify endpoint not available, proceeding anyway"
                    }
                    
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