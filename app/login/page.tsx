'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Eye, EyeOff, Mail, Lock } from 'lucide-react';

export default function LoginPage() {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      // TODO: Implement login logic with MongoDB
      console.log('Login attempt:', formData);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
    } catch (error) {
      console.error('Login error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGuestLogin = async (type: 'guest' | 'incognito') => {
    setIsLoading(true);
    
    try {
      console.log(`${type} login attempt`);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Set session storage for guest mode
      if (type === 'guest') {
        sessionStorage.setItem('userMode', 'guest');
        sessionStorage.setItem('guestId', `guest_${Date.now()}`);
      } else {
        sessionStorage.setItem('userMode', 'incognito');
        sessionStorage.setItem('incognitoId', `incognito_${Date.now()}`);
      }
      
      // Redirect to home
      window.location.href = 'https://proindra.github.io/prajwalindrakh-mlmondays/';
      
    } catch (error) {
      console.error(`${type} login error:`, error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Welcome Back</h1>
          <p className="auth-subtitle">Sign in to your Neural Insights account</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="email" className="form-label">
              <Mail size={18} />
              Email Address
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="form-input"
              placeholder="Enter your email"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password" className="form-label">
              <Lock size={18} />
              Password
            </label>
            <div className="password-input-container">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className="form-input"
                placeholder="Enter your password"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="password-toggle"
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="auth-button"
          >
            {isLoading ? 'Signing In...' : 'Sign In'}
          </button>

          <div className="auth-divider">
            <span>or</span>
          </div>

          <div className="guest-login-options">
            <button
              type="button"
              onClick={() => handleGuestLogin('guest')}
              className="guest-button"
              disabled={isLoading}
            >
              Continue as Guest
            </button>
          </div>
        </form>

        <div className="auth-footer">
          <p>
            Don't have an account?{' '}
            <Link href="/signup" className="auth-link">
              Create Account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}