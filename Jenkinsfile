pipeline {
    agent any
    environment {
        DOCKER_IMAGE = 'tiktok-socialscore-api'
        DOCKER_TAG = 'latest'
        API_URL = 'http://192.168.20.49:8000'  // Use your actual IP
        CALLBACK_URL = 'http://192.168.20.49:9696/customer/social-score'
        TEST_USERNAME = 'testuser'  // Replace with a real test username
    }
    stages {
        stage('Checkout') {
            steps { git url: 'https://github.com/yenat/Tik-Tok-socialscore.git', branch: 'main' }
        }
        
        stage('Build Docker') {
            steps { script { docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}") } }
        }
        
        stage('Deploy API') {
            steps {
                sh 'docker stop ${DOCKER_IMAGE} || true'
                sh 'docker rm ${DOCKER_IMAGE} || true'
                sh """
                docker run -d \
                    --name ${DOCKER_IMAGE} \
                    -p 8000:8000 \
                    ${DOCKER_IMAGE}:${DOCKER_TAG}
                """
                sleep(10)  // Wait for API to start
            }
        }
        
        stage('Test Verification Flow') {
            steps {
                script {
                    // 1. Generate verification code
                    def TEST_ID = "jenkins-test-${BUILD_NUMBER}"
                    def verifyResponse = sh(returnStdout: true, script: """
                        curl -s -X POST '${API_URL}/request-verification' \
                        -H 'Content-Type: application/json' \
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "type": "*SOCIAL_SCORE*",
                            "data": [{
                                "social_media": "TikTok",
                                "username": "${TEST_USERNAME}"
                            }]
                        }'
                    """).trim()
                    
                    logger.info("Verification response: ${verifyResponse}")
                    
                    // 2. Skip bio update (since we can't modify real accounts in CI)
                    // Normally you'd pause here and manually update the bio
                    
                    // 3. Verify with stored code (simulate success)
                    def storedCode = verification_storage[TEST_ID]?.code ?: "000000"
                    sh """
                        curl -v -X POST '${API_URL}/verify-and-score' \
                        -H 'Content-Type: application/json' \
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "type": "*SOCIAL_SCORE*",
                            "data": [{
                                "social_media": "TikTok",
                                "username": "${TEST_USERNAME}"
                            }],
                            "callbackUrl": "${CALLBACK_URL}"
                        }'
                    """
                }
            }
        }
    }
    post {
        always {
            sh 'docker logs ${DOCKER_IMAGE}'
            sh 'docker stop ${DOCKER_IMAGE} || true'
        }
    }
}