// Employee Attrition Dashboard JavaScript

// Data from provided JSON
const applicationData = {
  "employees": [
    {"id": "EMP001", "age": 35, "department": "Sales", "jobRole": "Sales Executive", "education": "Bachelor", "gender": "Male", "maritalStatus": "Married", "monthlyIncome": 4500, "distanceFromHome": 12, "yearsAtCompany": 8, "jobSatisfaction": 3, "workLifeBalance": 2, "overtime": "Yes", "attritionRisk": 0.762, "riskCategory": "High"},
    {"id": "EMP002", "age": 28, "department": "Research & Development", "jobRole": "Research Scientist", "education": "Master", "gender": "Female", "maritalStatus": "Single", "monthlyIncome": 6200, "distanceFromHome": 5, "yearsAtCompany": 3, "jobSatisfaction": 4, "workLifeBalance": 3, "overtime": "No", "attritionRisk": 0.245, "riskCategory": "Low"},
    {"id": "EMP003", "age": 42, "department": "Human Resources", "jobRole": "Manager", "education": "Master", "gender": "Female", "maritalStatus": "Married", "monthlyIncome": 8500, "distanceFromHome": 18, "yearsAtCompany": 12, "jobSatisfaction": 2, "workLifeBalance": 2, "overtime": "Yes", "attritionRisk": 0.834, "riskCategory": "High"},
    {"id": "EMP004", "age": 31, "department": "Sales", "jobRole": "Sales Representative", "education": "Bachelor", "gender": "Male", "maritalStatus": "Single", "monthlyIncome": 3200, "distanceFromHome": 25, "yearsAtCompany": 1, "jobSatisfaction": 1, "workLifeBalance": 1, "overtime": "Yes", "attritionRisk": 0.912, "riskCategory": "High"},
    {"id": "EMP005", "age": 39, "department": "Research & Development", "jobRole": "Laboratory Technician", "education": "College", "gender": "Male", "maritalStatus": "Married", "monthlyIncome": 2800, "distanceFromHome": 8, "yearsAtCompany": 15, "jobSatisfaction": 4, "workLifeBalance": 4, "overtime": "No", "attritionRisk": 0.156, "riskCategory": "Low"},
    {"id": "EMP006", "age": 26, "department": "Sales", "jobRole": "Sales Executive", "education": "Bachelor", "gender": "Female", "maritalStatus": "Single", "monthlyIncome": 4100, "distanceFromHome": 22, "yearsAtCompany": 0, "jobSatisfaction": 2, "workLifeBalance": 2, "overtime": "Yes", "attritionRisk": 0.887, "riskCategory": "High"},
    {"id": "EMP007", "age": 45, "department": "Research & Development", "jobRole": "Manufacturing Director", "education": "Doctor", "gender": "Male", "maritalStatus": "Married", "monthlyIncome": 12500, "distanceFromHome": 3, "yearsAtCompany": 18, "jobSatisfaction": 4, "workLifeBalance": 3, "overtime": "No", "attritionRisk": 0.178, "riskCategory": "Low"},
    {"id": "EMP008", "age": 33, "department": "Human Resources", "jobRole": "Human Resources", "education": "Bachelor", "gender": "Female", "maritalStatus": "Divorced", "monthlyIncome": 4800, "distanceFromHome": 16, "yearsAtCompany": 5, "jobSatisfaction": 1, "workLifeBalance": 3, "overtime": "Yes", "attritionRisk": 0.723, "riskCategory": "High"},
    {"id": "EMP009", "age": 29, "department": "Sales", "jobRole": "Manager", "education": "Master", "gender": "Male", "maritalStatus": "Single", "monthlyIncome": 7200, "distanceFromHome": 11, "yearsAtCompany": 6, "jobSatisfaction": 3, "workLifeBalance": 4, "overtime": "No", "attritionRisk": 0.334, "riskCategory": "Low"},
    {"id": "EMP010", "age": 37, "department": "Research & Development", "jobRole": "Research Scientist", "education": "Doctor", "gender": "Female", "maritalStatus": "Married", "monthlyIncome": 9800, "distanceFromHome": 14, "yearsAtCompany": 9, "jobSatisfaction": 2, "workLifeBalance": 2, "overtime": "Yes", "attritionRisk": 0.789, "riskCategory": "High"}
  ],
  "departmentStats": [
    {"department": "Sales", "totalEmployees": 47, "highRiskEmployees": 41, "avgAttritionRisk": 0.834, "avgMonthlySalary": 4250},
    {"department": "Research & Development", "totalEmployees": 63, "highRiskEmployees": 53, "avgAttritionRisk": 0.856, "avgMonthlySalary": 5680},
    {"department": "Human Resources", "totalEmployees": 40, "highRiskEmployees": 32, "avgAttritionRisk": 0.878, "avgMonthlySalary": 6100}
  ],
  "roiScenarios": [
    {"name": "Salary Increase (10%)", "description": "Increase salary by 10% for high-risk employees", "costPerEmployee": 6000, "riskReduction": 0.18, "applicableEmployees": 126},
    {"name": "Work-Life Program", "description": "Wellness and flex-time initiatives", "costPerEmployee": 2000, "riskReduction": 0.15, "applicableEmployees": 126},
    {"name": "Professional Development", "description": "Training and development opportunities", "costPerEmployee": 3000, "riskReduction": 0.17, "applicableEmployees": 126},
    {"name": "Comprehensive Package", "description": "Combination of salary, wellness, and training", "costPerEmployee": 8000, "riskReduction": 0.28, "applicableEmployees": 126}
  ],
  "featureImportance": [
    {"feature": "Overtime", "importance": 0.24},
    {"feature": "Monthly Income", "importance": 0.18},
    {"feature": "Age", "importance": 0.15},
    {"feature": "Distance From Home", "importance": 0.12},
    {"feature": "Job Satisfaction", "importance": 0.11},
    {"feature": "Years at Company", "importance": 0.09},
    {"feature": "Work Life Balance", "importance": 0.08},
    {"feature": "Education", "importance": 0.03}
  ],
  "overallStats": {
    "totalEmployees": 150,
    "highRiskEmployees": 126,
    "averageAttritionRisk": 0.856,
    "currentAttritionRate": 0.161
  }
};

// Generate employee names
const firstNames = ["John", "Jane", "Mike", "Sarah", "David", "Lisa", "Chris", "Emma", "Alex", "Maria", "Tom", "Anna", "Ryan", "Kate", "Mark", "Amy", "Paul", "Sophie", "Nick", "Laura", "James", "Emily", "Robert", "Jessica", "Michael", "Amanda", "Kevin", "Rachel", "Steven", "Michelle"];
const lastNames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee", "Gonzalez", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Perez", "Hall"];

// Add names to employees
applicationData.employees.forEach((employee, index) => {
  const firstName = firstNames[index % firstNames.length];
  const lastName = lastNames[Math.floor(index / firstNames.length) % lastNames.length];
  employee.name = `${firstName} ${lastName}`;
});

// Global variables
let filteredEmployees = [...applicationData.employees];
let charts = {};
let currentSortOrder = 'desc';

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
  initializeTabs();
  initializeFilters();
  updateDashboard();
  initializeEventListeners();
});

// Initialize tabs
function initializeTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  tabButtons.forEach(button => {
    button.addEventListener('click', function() {
      const targetTab = this.dataset.tab;
      
      // Remove active class from all tabs and panes
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabPanes.forEach(pane => pane.classList.remove('active'));
      
      // Add active class to clicked tab and corresponding pane
      this.classList.add('active');
      document.getElementById(targetTab).classList.add('active');
      
      // Load tab-specific content
      setTimeout(() => {
        if (targetTab === 'analytics') {
          renderAnalyticsCharts();
        }
      }, 100);
    });
  });
}

// Initialize filters
function initializeFilters() {
  const ageMin = document.getElementById('ageRangeMin');
  const ageMax = document.getElementById('ageRangeMax');
  
  // Set age range based on actual data
  const ages = applicationData.employees.map(emp => emp.age);
  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);
  
  ageMin.min = minAge;
  ageMin.max = maxAge;
  ageMax.min = minAge;
  ageMax.max = maxAge;
  ageMin.value = minAge;
  ageMax.value = maxAge;
  
  updateAgeRangeDisplay();
}

// Initialize event listeners
function initializeEventListeners() {
  // Filter controls
  document.getElementById('employeeSearch').addEventListener('input', applyFilters);
  document.getElementById('departmentFilter').addEventListener('change', applyFilters);
  document.getElementById('riskCategoryFilter').addEventListener('change', applyFilters);
  document.getElementById('ageRangeMin').addEventListener('input', function() {
    updateAgeRangeDisplay();
    applyFilters();
  });
  document.getElementById('ageRangeMax').addEventListener('input', function() {
    updateAgeRangeDisplay();
    applyFilters();
  });
  
  // Export and refresh buttons
  document.getElementById('exportBtn').addEventListener('click', exportReport);
  document.getElementById('refreshBtn').addEventListener('click', refreshPredictions);
}

// Update age range display
function updateAgeRangeDisplay() {
  const minAge = document.getElementById('ageRangeMin').value;
  const maxAge = document.getElementById('ageRangeMax').value;
  document.getElementById('ageRangeDisplay').textContent = `${minAge}-${maxAge}`;
}

// Update dashboard
function updateDashboard() {
  updateStatsCards();
  updateDepartmentBreakdown();
  renderRiskOverviewCharts();
  populateEmployeeTable();
}

// Update stats cards
function updateStatsCards() {
  document.getElementById('totalEmployees').textContent = applicationData.overallStats.totalEmployees;
  document.getElementById('highRiskCount').textContent = applicationData.overallStats.highRiskEmployees;
  document.getElementById('avgRiskScore').textContent = `${(applicationData.overallStats.averageAttritionRisk * 100).toFixed(1)}%`;
  document.getElementById('currentAttritionRate').textContent = `${(applicationData.overallStats.currentAttritionRate * 100).toFixed(1)}%`;
}

// Update department breakdown table
function updateDepartmentBreakdown() {
  const tableBody = document.getElementById('departmentTableBody');
  tableBody.innerHTML = '';
  
  applicationData.departmentStats.forEach(dept => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${dept.department}</td>
      <td>${dept.totalEmployees}</td>
      <td>${dept.highRiskEmployees}</td>
      <td>${(dept.avgAttritionRisk * 100).toFixed(1)}%</td>
      <td>$${dept.avgMonthlySalary.toLocaleString()}</td>
    `;
    tableBody.appendChild(row);
  });
}

// Render risk overview charts
function renderRiskOverviewCharts() {
  renderDepartmentRiskChart();
}

// Department risk chart
function renderDepartmentRiskChart() {
  const ctx = document.getElementById('departmentRiskChart').getContext('2d');
  
  if (charts.departmentRisk) {
    charts.departmentRisk.destroy();
  }
  
  const departments = applicationData.departmentStats.map(dept => dept.department);
  const riskData = applicationData.departmentStats.map(dept => dept.avgAttritionRisk * 100);
  
  charts.departmentRisk = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: departments,
      datasets: [{
        label: 'Average Risk %',
        data: riskData,
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: 'Risk Percentage'
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

// Apply filters to employee table
function applyFilters() {
  const searchTerm = document.getElementById('employeeSearch').value.toLowerCase();
  const departmentFilter = document.getElementById('departmentFilter').value;
  const riskCategoryFilter = document.getElementById('riskCategoryFilter').value;
  const minAge = parseInt(document.getElementById('ageRangeMin').value);
  const maxAge = parseInt(document.getElementById('ageRangeMax').value);
  
  filteredEmployees = applicationData.employees.filter(emp => {
    const matchesSearch = !searchTerm || 
      emp.name.toLowerCase().includes(searchTerm) || 
      emp.id.toLowerCase().includes(searchTerm);
    const matchesDepartment = !departmentFilter || emp.department === departmentFilter;
    const matchesRiskCategory = !riskCategoryFilter || emp.riskCategory === riskCategoryFilter;
    const matchesAge = emp.age >= minAge && emp.age <= maxAge;
    
    return matchesSearch && matchesDepartment && matchesRiskCategory && matchesAge;
  });
  
  populateEmployeeTable();
}

// Populate employee table
function populateEmployeeTable() {
  const tableBody = document.getElementById('employeeTableBody');
  tableBody.innerHTML = '';
  
  // Sort by risk score (descending by default)
  const sortedEmployees = [...filteredEmployees].sort((a, b) => {
    return currentSortOrder === 'desc' ? b.attritionRisk - a.attritionRisk : a.attritionRisk - b.attritionRisk;
  });
  
  sortedEmployees.forEach(emp => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${emp.id}</td>
      <td>${emp.name}</td>
      <td>${emp.department}</td>
      <td>${emp.jobRole}</td>
      <td>${emp.age}</td>
      <td>${(emp.attritionRisk * 100).toFixed(1)}%</td>
      <td><span class="risk-${emp.riskCategory.toLowerCase()}">${emp.riskCategory}</span></td>
    `;
    tableBody.appendChild(row);
  });
}

// Sort by risk score
function sortByRisk() {
  currentSortOrder = currentSortOrder === 'desc' ? 'asc' : 'desc';
  populateEmployeeTable();
}

// Run scenario simulation
function runScenarioSimulation(scenarioIndex) {
  const scenario = applicationData.roiScenarios[scenarioIndex];
  const button = document.querySelectorAll('.scenario-btn')[scenarioIndex];
  
  // Add loading state
  button.classList.add('loading');
  
  setTimeout(() => {
    // Calculate ROI
    const totalInvestment = scenario.costPerEmployee * scenario.applicableEmployees;
    const employeesRetained = Math.round(scenario.applicableEmployees * scenario.riskReduction);
    const avgSalary = 75000; // Average annual salary
    const costSavings = employeesRetained * (avgSalary * 1.5); // 150% of salary for replacement cost
    const netROI = ((costSavings - totalInvestment) / totalInvestment) * 100;
    
    // Update results
    document.getElementById('totalInvestment').textContent = `$${totalInvestment.toLocaleString()}`;
    document.getElementById('employeesRetained').textContent = employeesRetained;
    document.getElementById('costSavings').textContent = `$${costSavings.toLocaleString()}`;
    document.getElementById('netROI').textContent = `${netROI.toFixed(1)}%`;
    
    // Show results
    document.getElementById('simulationResults').style.display = 'block';
    
    // Render ROI comparison chart
    renderROIComparisonChart(scenario.name, totalInvestment, costSavings, netROI);
    
    button.classList.remove('loading');
  }, 1500);
}

// Render ROI comparison chart
function renderROIComparisonChart(scenarioName, investment, savings, roi) {
  const ctx = document.getElementById('roiComparisonChart').getContext('2d');
  
  if (charts.roiComparison) {
    charts.roiComparison.destroy();
  }
  
  charts.roiComparison = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Investment Cost', 'Cost Savings', 'Net Benefit'],
      datasets: [{
        label: 'Amount ($)',
        data: [investment, savings, savings - investment],
        backgroundColor: ['#B4413C', '#1FB8CD', savings - investment > 0 ? '#5D878F' : '#B4413C']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString();
            }
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: `${scenarioName} - Financial Impact`
        },
        legend: {
          display: false
        }
      }
    }
  });
}

// Render analytics charts
function renderAnalyticsCharts() {
  renderFeatureImportanceChart();
  renderRiskDistributionChart();
  renderTrendChart();
}

// Feature importance chart
function renderFeatureImportanceChart() {
  const ctx = document.getElementById('featureImportanceChart').getContext('2d');
  
  if (charts.featureImportance) {
    charts.featureImportance.destroy();
  }
  
  const features = applicationData.featureImportance.map(item => item.feature);
  const importance = applicationData.featureImportance.map(item => item.importance * 100);
  
  charts.featureImportance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: features,
      datasets: [{
        label: 'Importance (%)',
        data: importance,
        backgroundColor: '#ECEBD5'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: {
          beginAtZero: true,
          max: 30,
          title: {
            display: true,
            text: 'Importance (%)'
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

// Risk distribution pie chart
function renderRiskDistributionChart() {
  const ctx = document.getElementById('riskDistributionChart').getContext('2d');
  
  if (charts.riskDistribution) {
    charts.riskDistribution.destroy();
  }
  
  const riskCounts = {
    'Low': applicationData.employees.filter(emp => emp.riskCategory === 'Low').length,
    'Medium': applicationData.employees.filter(emp => emp.riskCategory === 'Medium').length,
    'High': applicationData.employees.filter(emp => emp.riskCategory === 'High').length
  };
  
  charts.riskDistribution = new Chart(ctx, {
    type: 'pie',
    data: {
      labels: ['Low Risk', 'Medium Risk', 'High Risk'],
      datasets: [{
        data: [riskCounts.Low, riskCounts.Medium, riskCounts.High],
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom'
        }
      }
    }
  });
}

// Trend chart (mock data)
function renderTrendChart() {
  const ctx = document.getElementById('trendChart').getContext('2d');
  
  if (charts.trend) {
    charts.trend.destroy();
  }
  
  // Generate mock monthly trend data
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const trendData = [0.89, 0.87, 0.85, 0.88, 0.86, 0.84, 0.87, 0.85, 0.86, 0.84, 0.85, 0.86];
  
  charts.trend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: months,
      datasets: [{
        label: 'Average Attrition Risk',
        data: trendData.map(val => val * 100),
        borderColor: '#1FB8CD',
        backgroundColor: 'rgba(31, 184, 205, 0.1)',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          min: 80,
          max: 95,
          title: {
            display: true,
            text: 'Risk Percentage (%)'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Month'
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

// Export report
function exportReport() {
  const reportData = {
    timestamp: new Date().toISOString(),
    summary: applicationData.overallStats,
    departmentBreakdown: applicationData.departmentStats,
    highRiskEmployees: applicationData.employees.filter(emp => emp.riskCategory === 'High').map(emp => ({
      id: emp.id,
      name: emp.name,
      department: emp.department,
      jobRole: emp.jobRole,
      riskScore: (emp.attritionRisk * 100).toFixed(1) + '%'
    }))
  };
  
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reportData, null, 2));
  const downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", `attrition_report_${new Date().toISOString().split('T')[0]}.json`);
  document.body.appendChild(downloadAnchorNode);
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
}

// Refresh predictions
function refreshPredictions() {
  const button = document.getElementById('refreshBtn');
  button.classList.add('loading');
  
  setTimeout(() => {
    // Simulate prediction refresh by slightly adjusting risk scores
    applicationData.employees.forEach(emp => {
      const adjustment = (Math.random() - 0.5) * 0.1; // ±5% adjustment
      emp.attritionRisk = Math.max(0.05, Math.min(0.95, emp.attritionRisk + adjustment));
      
      // Update risk category based on new score
      if (emp.attritionRisk >= 0.7) emp.riskCategory = "High";
      else if (emp.attritionRisk >= 0.4) emp.riskCategory = "Medium";
      else emp.riskCategory = "Low";
    });
    
    // Update overall stats
    const highRiskCount = applicationData.employees.filter(emp => emp.riskCategory === 'High').length;
    const avgRisk = applicationData.employees.reduce((sum, emp) => sum + emp.attritionRisk, 0) / applicationData.employees.length;
    
    applicationData.overallStats.highRiskEmployees = highRiskCount;
    applicationData.overallStats.averageAttritionRisk = avgRisk;
    
    filteredEmployees = [...applicationData.employees];
    updateDashboard();
    button.classList.remove('loading');
  }, 2000);
}