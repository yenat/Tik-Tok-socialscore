pipeline {
    agent any
    environment {
        DOCKER_IMAGE = 'tiktok-socialscore-api'
        DOCKER_TAG = 'latest'
        // Test variables (customize these)
        TEST_API_URL = 'http://localhost:8000'  // Uses Jenkins' local network
        TEST_CALLBACK_URL = 'http://192.168.20.49:9696/customer/credit-score'
        TEST_USERNAME = 'testuser'  // Use a real test TikTok username
    }
    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/yenat/Tik-Tok-socialscore.git',
                   branch: 'main'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }
        stage('Deploy API') {
            steps {
                script {
                    sh 'docker stop ${DOCKER_IMAGE} || true'
                    sh 'docker rm ${DOCKER_IMAGE} || true'
                    sh """
                    docker run -d \
                        --name ${DOCKER_IMAGE} \
                        -p 8000:8000 \
                        -e PORT=8000 \
                        -e MODEL_PATH=/app/tiktok_scoring_model.pkl \
                        -e SCALING_PARAMS_PATH=/app/scaling_params.pkl \
                        ${DOCKER_IMAGE}:${DOCKER_TAG}
                    """
                }
            }
        }
        stage('Test Callback Flow') {
            steps {
                script {
                    // Generate unique test ID
                    def TEST_ID = "jenkins-test-${BUILD_NUMBER}"
                    // 1. Trigger API with callback
                    sh """
                    curl -X POST '${TEST_API_URL}/verify-and-score' \
                    -H 'Content-Type: application/json' \
                    -d '{
                        "fayda_number": "${TEST_ID}",
                        "type": "*SOCIAL_SCORE",
                        "data": [{
                            "social_media": "TikTok",
                            "username": "${TEST_USERNAME}"
                        }],
                        "callbackUrl": "${TEST_CALLBACK_URL}"
                    }'
                    """
                    // 2. Verify callback (simplified - adapt to your callback server)
                    // Option A: If you have callback logs (recommended)
                    sh """
                    echo "Waiting 10s for callback..."
                    sleep 10
                    curl -s 'http://192.168.20.49:9696/logs' | grep -q "${TEST_ID}" || {
                        echo ":x: Callback not received";
                        exit 1;
                    }
                    echo ":white_check_mark: Callback verified for ${TEST_ID}"
                    """
                    // Option B: If no logs, just verify API response (fallback)
                    // sh 'curl -s "${TEST_API_URL}/health" | grep -q "healthy"'
                }
            }
        }
    }
    post {
        always {
            sh 'docker system prune -f'
        }
        success {
            echo 'Deployment AND callback test succeeded'
        }
        failure {
            echo 'Check callback server at http://192.168.20.49:9696/logs'
        }
    }
}