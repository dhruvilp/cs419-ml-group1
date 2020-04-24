export const authService = {
    login,
    signup,
    logout
};

const BASE_URL = 'http://127.0.0.1:5000';

async function login(username, password) {
    const requestOptions = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            "username" : username, 
            "password" : password 
        })
    };

    const response = await fetch(`${BASE_URL}/login`, requestOptions);
    const user = await handleResponse(response);
    if (user) {
        localStorage.setItem('user', JSON.stringify(user));
    }
    return user;
}

async function signup(username, password) {
    const requestOptions = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            "username" : username, 
            "password" : password 
        })
    };

    const response = await fetch(`${BASE_URL}/signup`, requestOptions);
    const user = await handleResponse(response);
    if (user.ok) {
        localStorage.setItem('user', JSON.stringify(user));
    }
    return user;
}

function logout() {
    localStorage.removeItem('user');
}

function handleResponse(response) {
    return response.text().then(text => {
        const data = (response.ok) && text && JSON.parse(text);
        if (!response.ok) {
            if (response.status === 401) {
                // auto logout if 401 response returned from api
                logout();
                // location.reload(true);
            }
            const error = (data && data.message) || response.statusText;
            return Promise.reject(error);
        }
        return data;
    });
}