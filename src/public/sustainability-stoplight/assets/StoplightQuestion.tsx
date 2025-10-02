import { useState } from 'react';
import { StimulusParams } from '../../../store/types';

interface StoplightOption {
  color: string;
  value: string;
  label: string;
  icon?: string;
}

interface StoplightParams {
  questionId: string;
  questionText: string;
  options: StoplightOption[];
}

// This component creates a stoplight-style question display
// mimicking the sustainability stoplight cards shown in the images
function StoplightQuestion({ parameters, setAnswer }: StimulusParams<StoplightParams>) {
  const [selectedOption, setSelectedOption] = useState<string>('');

  const { questionId, questionText, options } = parameters;

  // Check if all options have the same color (neutral question)
  const isNeutralQuestion = options.every((opt) => opt.color === options[0].color);

  // Handle option selection
  const handleSelect = (value: string, _color: string) => {
    setSelectedOption(value);
    setAnswer({
      status: true,
      answers: {
        [questionId]: value,
      },
    });
  };

  // Map color codes to actual colors
  const getColorStyles = (color: string) => {
    // For neutral questions (all same color), use blue theme instead of green
    if (isNeutralQuestion && color === 'green') {
      return {
        bg: '#2196F3',
        border: '#1976D2',
        light: '#E3F2FD',
      };
    }

    const colors = {
      red: {
        bg: '#E85D4F',
        border: '#C44336',
        light: '#FFEBEE',
      },
      yellow: {
        bg: '#F9D949',
        border: '#F9C70E',
        light: '#FFFDE7',
      },
      green: {
        bg: '#72B856',
        border: '#5C9A43',
        light: '#E8F5E9',
      },
    };
    return colors[color as keyof typeof colors] || colors.green;
  };

  return (
    <div style={{
      maxWidth: '1000px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: 'Arial, sans-serif',
    }}
    >
      {/* Question Title */}
      <h2 style={{
        fontSize: '24px',
        marginBottom: '40px',
        textAlign: 'center',
        color: '#333',
      }}
      >
        {questionText}
      </h2>

      {/* Stoplight Cards Container */}
      <div style={{
        display: 'flex',
        gap: '20px',
        justifyContent: 'center',
        flexWrap: 'wrap',
      }}
      >
        {options.map((option, index: number) => {
          const colorStyles = getColorStyles(option.color);
          const isSelected = selectedOption === option.value;

          return (
            <div
              key={index}
              onClick={() => handleSelect(option.value, option.color)}
              style={{
                flex: '1',
                minWidth: '280px',
                maxWidth: '320px',
                height: '480px',
                cursor: 'pointer',
                border: isSelected ? `4px solid ${colorStyles.border}` : '2px solid #ddd',
                borderRadius: '12px',
                overflow: 'hidden',
                boxShadow: isSelected
                  ? `0 8px 16px rgba(0, 0, 0, 0.2), 0 0 0 4px ${colorStyles.light}`
                  : '0 4px 8px rgba(0, 0, 0, 0.1)',
                transition: 'all 0.3s ease',
                transform: isSelected ? 'scale(1.05)' : 'scale(1)',
                backgroundColor: 'white',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              {/* Stoplight Indicator */}
              <div style={{
                height: '80px',
                backgroundColor: colorStyles.bg,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
              }}
              >
                {/* Stoplight Circle */}
                <div style={{
                  width: '50px',
                  height: '50px',
                  borderRadius: '50%',
                  backgroundColor: 'white',
                  border: `3px solid ${colorStyles.border}`,
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
                }}
                >
                  {/* Inner colored circle */}
                  <div style={{
                    width: '100%',
                    height: '100%',
                    borderRadius: '50%',
                    backgroundColor: colorStyles.bg,
                    transform: 'scale(0.7)',
                  }}
                  />
                </div>
              </div>

              {/* Illustration Area (placeholder for now) */}
              <div style={{
                height: '200px',
                backgroundColor: '#f5f5f5',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderTop: `3px solid ${colorStyles.border}`,
                borderBottom: `3px solid ${colorStyles.border}`,
              }}
              >
                {/* Placeholder for illustration */}
                <div style={{
                  fontSize: '60px',
                  opacity: 0.3,
                }}
                >
                  {option.icon || (option.color === 'red' ? '🚿' : option.color === 'yellow' ? '🚿🚿' : '🚿🚿🚿')}
                </div>
              </div>

              {/* Option Label/Description */}
              <div style={{
                padding: '20px',
                backgroundColor: 'white',
                minHeight: '120px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              >
                <p style={{
                  margin: 0,
                  fontSize: '16px',
                  textAlign: 'center',
                  color: '#333',
                  fontWeight: isSelected ? 'bold' : 'normal',
                  lineHeight: '1.5',
                }}
                >
                  {option.label}
                </p>
              </div>

              {/* Footer Label */}
              <div style={{
                padding: '12px',
                backgroundColor: colorStyles.bg,
                textAlign: 'center',
              }}
              >
                <span style={{
                  fontSize: '18px',
                  fontWeight: 'bold',
                  color: 'white',
                  textTransform: 'capitalize',
                }}
                >
                  {isNeutralQuestion ? option.label : option.color}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selection indicator */}
      {selectedOption && (
        <div style={{
          marginTop: '30px',
          padding: '15px',
          backgroundColor: '#e8f5e9',
          border: '2px solid #4caf50',
          borderRadius: '8px',
          textAlign: 'center',
        }}
        >
          <p style={{ margin: 0, color: '#2e7d32', fontWeight: 'bold' }}>
            ✓ Selection recorded
          </p>
        </div>
      )}
    </div>
  );
}

export default StoplightQuestion;
