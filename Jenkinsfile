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
            
            // Start container with logs visible
            sh """
            docker run -d \
                --name ${DOCKER_IMAGE} \
                -p 8000:8000 \
                ${DOCKER_IMAGE}:${DOCKER_TAG}
            """
            
            // Wait for container to start
            sleep(10)
            
            // Check container status
            def status = sh(returnStdout: true, script: "docker inspect -f '{{.State.Status}}' ${DOCKER_IMAGE}").trim()
            if (status != "running") {
                error("Container failed to start. Status: ${status}")
            }
            
            // Get container logs for debugging
            sh "docker logs ${DOCKER_IMAGE} > container_logs.txt 2>&1"
            archiveArtifacts artifacts: 'container_logs.txt'
            
            // Verify API health
            try {
                sh "curl -v --retry 5 --retry-delay 5 --max-time 10 http://localhost:8000/health"
            } catch (err) {
                error("API health check failed after container started")
            }
        }
    }
}
        
        stage('Test API') {
            steps {
                script {
                    // 1. Health Check
                    sh 'curl -f ${API_URL}/health'
                    
                    // 2. Quick Test (bypasses verification)
                    sh """
                        curl -v -X POST '${API_URL}/verify-and-score' \\
                        -H 'Content-Type: application/json' \\
                        -d '{
                            "fayda_number": "jenkins-test-${BUILD_NUMBER}",
                            "type": "*SOCIAL_SCORE*",
                            "data": [{
                                "social_media": "TikTok", 
                                "username": "${TEST_USERNAME}"
                            }],
                            "callbackUrl": "${CALLBACK_URL}"
                        }' || echo "Verification failed (expected without bio update)"
                    """
                    
                    // 3. Full Verification Flow
                    def TEST_ID = "jenkins-fulltest-${BUILD_NUMBER}"
                    sh """
                        curl -v -X POST '${API_URL}/request-verification' \\
                        -H 'Content-Type: application/json' \\
                        -d '{
                            "fayda_number": "${TEST_ID}",
                            "type": "*SOCIAL_SCORE*",
                            "data": [{
                                "social_media": "TikTok",
                                "username": "${TEST_USERNAME}"
                            }]
                        }'
                    """
                    echo "Verification code generated for ${TEST_ID}"
                }
            }
        }
    }
    post {
        always {
            sh 'docker logs ${DOCKER_IMAGE} --tail 50'
            sh 'docker stop ${DOCKER_IMAGE} || true'
        }
    }
}