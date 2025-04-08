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
            steps { git url: 'https://github.com/yenat/Tik-Tok-socialscore.git', branch: 'main' }
        }
        
        stage('Build Docker') {
            steps { script { docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}") } }
        }
        
        stage('Deploy API') {
            steps {
                script {
                    sh 'docker stop ${DOCKER_IMAGE} || true'
                    sh 'docker rm ${DOCKER_IMAGE} || true'
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
                    if (!healthy) error("API failed to start")
                }
            }
        }
        
        stage('Test API') {
            steps {
                script {
                    // Install jq if missing
                    sh 'command -v jq || (apt-get update && apt-get install -y jq) || true'
                    
                    def TEST_ID = "jenkins-test-${BUILD_NUMBER}"
                    
                    // 1. Request verification code
                    def verification = sh(returnStdout: true, script: """
                        curl -s -X POST ${API_URL}/request-verification \
                        -H 'Content-Type: application/json' \
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "type": "*SOCIAL_SCORE*",
                            "data": [{
                                "social_media": "TikTok",
                                "username": "${TEST_USERNAME}"
                            }]
                        }'
                    """)
                    
                    // 2. Skip mock verification (since endpoint doesn't exist)
                    // Instead we'll manually verify by adding code to test storage
                    def verificationCode = sh(returnStdout: true, 
                        script: "echo '${verification}' | jq -r '.verification_code'").trim()
                    
                    // 3. Verify and get score (with automatic verification bypass)
                    def score = sh(returnStdout: true, script: """
                        curl -s -X POST ${API_URL}/verify-and-score \
                        -H 'Content-Type: application/json' \
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "type": "*SOCIAL_SCORE*", 
                            "data": [{
                                "social_media": "TikTok",
                                "username": "${TEST_USERNAME}"
                            }],
                            "callbackUrl": "${CALLBACK_URL}",
                            "test_mode": true,
                            "verification_code": "${verificationCode}"
                        }'
                    """)
                    
                    // Validate score response
                    if (!score.contains('socialscore')) {
                        error("Invalid score response: ${score}")
                    }
                    echo "Score test passed: ${score}"
                }
            }
        }
    }
    post {
        always {
            script {
                sh 'docker logs ${DOCKER_IMAGE} --tail 50'
                sh 'docker stop ${DOCKER_IMAGE} || true'
            }
        }
    }
}